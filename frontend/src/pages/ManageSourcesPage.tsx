import { useEffect, useState } from "react";
import { api, CompanySuggestion, Favorite, JobBoardSuggestion, Source, SourceKind } from "../api";
import { Spinner } from "../components/Spinner";

const KIND_LABELS: Record<SourceKind, string> = {
  ats_greenhouse: "Greenhouse (company board token)",
  ats_lever: "Lever (company slug)",
  ats_ashby: "Ashby (board name)",
  aggregator_adzuna: "Adzuna (keyword search)",
  aggregator_remoteok: "RemoteOK (tag filter)",
  custom_board: "Custom board (any URL)",
};

const CONFIG_FIELD: Record<SourceKind, { key: string; label: string; placeholder: string } | null> = {
  ats_greenhouse: {
    key: "board_token",
    label: "Job board URL or token",
    placeholder: "https://boards.greenhouse.io/stripe",
  },
  ats_lever: {
    key: "company_slug",
    label: "Job board URL or slug",
    placeholder: "https://jobs.lever.co/ro",
  },
  ats_ashby: {
    key: "board_name",
    label: "Job board URL or name",
    placeholder: "https://jobs.ashbyhq.com/linear",
  },
  aggregator_adzuna: { key: "what", label: "Search keyword", placeholder: "e.g. software engineer" },
  aggregator_remoteok: { key: "tags", label: "Tags (optional)", placeholder: "e.g. python,backend" },
  custom_board: { key: "url", label: "Job board URL", placeholder: "https://blueroseresearch.org/jobs/" },
};

const URL_PATTERNS: { kind: SourceKind; configKey: string; pattern: RegExp }[] = [
  { kind: "ats_greenhouse", configKey: "board_token", pattern: /(?:boards|job-boards)\.greenhouse\.io\/([a-z0-9_-]+)/i },
  { kind: "ats_lever", configKey: "company_slug", pattern: /jobs\.lever\.co\/([a-z0-9_-]+)/i },
  { kind: "ats_ashby", configKey: "board_name", pattern: /jobs\.ashbyhq\.com\/([a-z0-9_-]+)/i },
];

function detectSourceFromUrl(input: string): { kind: SourceKind; config: Record<string, string> } | null {
  const trimmed = input.trim();
  if (!/^https?:\/\//i.test(trimmed)) return null;
  for (const { kind, configKey, pattern } of URL_PATTERNS) {
    const match = trimmed.match(pattern);
    if (match) return { kind, config: { [configKey]: match[1] } };
  }
  return { kind: "custom_board", config: { url: trimmed } };
}

function FavoriteCompaniesSection() {
  const [favorites, setFavorites] = useState<Favorite[]>([]);
  const [name, setName] = useState("");
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = () => api.listFavorites().then(setFavorites);

  useEffect(() => {
    refresh();
  }, []);

  const detected = detectSourceFromUrl(url);

  const add = async () => {
    if (!name.trim() || !url.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await api.addFavorite(name.trim(), detected ? { kind: detected.kind, config: detected.config } : undefined);
      setName("");
      setUrl("");
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
        Any new posting from a favorite company always gets scored and flagged, regardless of the
        daily cap or your exclude-keyword rules.
      </p>
      <div className="form-row">
        <input
          placeholder="Company name, e.g. Anthropic"
          value={name}
          onChange={(e) => setName(e.target.value)}
          style={{ flex: "1 1 200px" }}
        />
        <input
          placeholder="Job board URL, e.g. https://jobs.lever.co/anthropic"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          style={{ flex: "2 1 280px" }}
        />
        <button className="primary" onClick={add} disabled={busy}>
          {busy ? <Spinner label="Adding..." /> : "Add favorite"}
        </button>
      </div>
      {detected && <p className="muted small">Detected: {KIND_LABELS[detected.kind]}</p>}
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
      {favorites.length === 0 && <p className="empty-state">No favorite companies yet.</p>}
    </div>
  );
}

function FavoriteBoardsSection({ sources, onChanged }: { sources: Source[]; onChanged: () => void }) {
  const [name, setName] = useState("");
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const detected = detectSourceFromUrl(url);
  const boards = sources.filter((s) => s.is_favorite && !s.company_name);

  const add = async () => {
    if (!name.trim() || !detected) return;
    setBusy(true);
    setError(null);
    try {
      await api.createSource({
        kind: detected.kind,
        name: name.trim(),
        config: detected.config,
        is_favorite: true,
      });
      setName("");
      setUrl("");
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const unfavorite = async (id: number) => {
    await api.updateSource(id, { is_favorite: false });
    onChanged();
  };

  return (
    <div className="card">
      <h2>
        <span className="star">★</span> Favorite job boards
      </h2>
      <p className="muted small">
        Any new posting from a favorite board always gets scored and flagged, regardless of the
        daily cap or your exclude-keyword rules.
      </p>
      <div className="form-row">
        <input
          placeholder="Board name, e.g. Tech Jobs for Good"
          value={name}
          onChange={(e) => setName(e.target.value)}
          style={{ flex: "1 1 200px" }}
        />
        <input
          placeholder="Board URL, e.g. https://www.techjobsforgood.com/"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          style={{ flex: "2 1 280px" }}
        />
        <button className="primary" onClick={add} disabled={busy || !detected}>
          {busy ? <Spinner label="Adding..." /> : "Add favorite"}
        </button>
      </div>
      {detected && <p className="muted small">Detected: {KIND_LABELS[detected.kind]}</p>}
      {error && <p className="error-text small">{error}</p>}

      <ul>
        {boards.map((s) => (
          <li key={s.id}>
            {s.name} — <code className="path">{JSON.stringify(s.config)}</code>
            {" — "}
            <button className="link small" onClick={() => unfavorite(s.id)}>
              remove
            </button>
          </li>
        ))}
      </ul>
      {boards.length === 0 && <p className="empty-state">No favorite job boards yet.</p>}
    </div>
  );
}

function DiscoverCompaniesSection() {
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
      <h2>Discover companies</h2>
      <p className="muted small">
        Uses your candidate profile and desired role (Profile page), plus a web search, to suggest
        companies worth monitoring. Nothing is added until you approve it.
      </p>
      <button className="ghost" onClick={runDiscovery} disabled={busy}>
        {busy ? <Spinner label="Searching..." /> : "Suggest companies"}
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

function DiscoverJobBoardsSection() {
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

  const approve = async (board: JobBoardSuggestion) => {
    setStatus((s) => ({ ...s, [board.url]: "working" }));
    try {
      await api.createSource({
        kind: "custom_board",
        name: board.name,
        config: { url: board.url },
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
        multi-employer boards worth monitoring — broader reach than tracking one company at a
        time. Nothing is added until you approve it.
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
              <button className="ghost small" onClick={() => approve(s)}>
                Monitor
              </button>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function AddSourceForm({ onAdded }: { onAdded: (source: Source) => void }) {
  const [kind, setKind] = useState<SourceKind>("ats_greenhouse");
  const [name, setName] = useState("");
  const [configValue, setConfigValue] = useState("");
  const [companyName, setCompanyName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const field = CONFIG_FIELD[kind];
  const detected = detectSourceFromUrl(configValue);

  const handleConfigChange = (value: string) => {
    setConfigValue(value);
    const match = detectSourceFromUrl(value);
    if (match) setKind(match.kind);
  };

  const submit = async () => {
    if (!name.trim() || !configValue.trim() || !field) return;
    setBusy(true);
    setError(null);
    try {
      const config =
        detected?.config ??
        (field.key === "tags"
          ? { tags: configValue.split(",").map((t) => t.trim()).filter(Boolean) }
          : { [field.key]: configValue.trim() });
      const source = await api.createSource({
        kind: detected?.kind ?? kind,
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
      <p className="muted small">
        Track a company's job board or an aggregator — any new postings feed into the All Jobs
        page and the daily digest.
      </p>
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
        <span className="flabel">{field?.label ?? "Value"}</span>
        <input
          placeholder={field?.placeholder}
          value={configValue}
          onChange={(e) => handleConfigChange(e.target.value)}
          style={{ width: "100%" }}
        />
        {detected && (
          <p className="muted small">
            Detected: {KIND_LABELS[detected.kind]} —{" "}
            {Object.entries(detected.config)
              .map(([k, v]) => `${k}: ${v}`)
              .join(", ")}
          </p>
        )}
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
          {busy ? <Spinner label="Testing..." /> : "Test fetch"}
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

export default function ManageSourcesPage() {
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

  const trackedSources = sources.filter((s) => !s.is_favorite);

  return (
    <div>
      <h1>Manage Sources</h1>
      <p className="muted" style={{ marginBottom: "18px" }}>
        Favorite companies/boards, discover new ones, and track whatever else you want polled. See
        the All Jobs page for what's turned up. For LinkedIn/Indeed/Glassdoor or anything you found
        yourself, use "Add a job" on the My Jobs page instead.
      </p>

      <FavoriteCompaniesSection />
      <FavoriteBoardsSection sources={sources} onChanged={refresh} />

      <DiscoverCompaniesSection />
      <DiscoverJobBoardsSection />

      <AddSourceForm onAdded={() => refresh()} />

      {trackedSources.map((s) => (
        <SourceCard key={s.id} source={s} onChanged={refresh} />
      ))}
      {trackedSources.length === 0 && (
        <p className="empty-state">No other tracked sources yet — add one above.</p>
      )}

      <div className="card">
        <h2>Run the daily pipeline now</h2>
        <p className="muted small">
          In production this runs automatically via Cloud Scheduler. Locally, trigger it manually
          to test.
        </p>
        <button className="ghost" onClick={runNow} disabled={running}>
          {running ? <Spinner label="Running..." /> : "Run now"}
        </button>
        {runResult && (
          <pre style={{ whiteSpace: "pre-wrap", marginTop: "10px" }}>
            <code>{runResult}</code>
          </pre>
        )}
      </div>
    </div>
  );
}
