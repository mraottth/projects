import { useEffect, useState } from "react";
import { api, Source, SourceKind } from "../api";
import { DiscoverCompaniesSection, DiscoverJobBoardsSection, DiscoverySeed } from "../components/Discovery";
import { detectSourceFromUrl } from "../sourceUrl";
import { Spinner } from "../components/Spinner";

const KIND_LABELS: Record<SourceKind, string> = {
  ats_greenhouse: "Greenhouse (organization board token)",
  ats_lever: "Lever (organization slug)",
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
        Track an organization's job board or an aggregator — any new postings feed into the All
        Jobs page and the daily digest.
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
        <span className="flabel">Organization display name (optional)</span>
        <input
          placeholder="Organization display name"
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

// Organizations vs. job boards, for the page's two sections: a single employer's ATS board, or
// any source with an organization name, is an organization; multi-employer boards and the
// keyword aggregators (no single employer) are job boards.
function isOrganization(source: Source): boolean {
  return source.kind.startsWith("ats_") || !!source.company_name;
}

// What "find more like this" asks discovery for: the name, plus the URL when there is one.
function describeForDiscovery(source: Source): string {
  const url = typeof source.config.url === "string" ? source.config.url : null;
  return url ? `${source.name} (${url})` : source.name;
}

function SourceCard({
  source,
  onChanged,
  onFindMoreLikeThis,
}: {
  source: Source;
  onChanged: () => void;
  onFindMoreLikeThis: (source: Source) => void;
}) {
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
        <button className="link small" onClick={() => onFindMoreLikeThis(source)}>
          Find more like this
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
  const [orgSeed, setOrgSeed] = useState<DiscoverySeed | null>(null);
  const [boardSeed, setBoardSeed] = useState<DiscoverySeed | null>(null);

  const refresh = () => api.listSources().then(setSources);

  // Starts a "more like this" search in the matching Discover card, and scrolls to it so the
  // results are where the user's looking.
  const findMoreLikeThis = (source: Source) => {
    const seed = { like: describeForDiscovery(source), label: source.name, nonce: Date.now() };
    const org = isOrganization(source);
    (org ? setOrgSeed : setBoardSeed)(seed);
    document
      .getElementById(org ? "discover-organizations" : "discover-job-boards")
      ?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

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


  const organizations = sources.filter(isOrganization);
  const boards = sources.filter((s) => !isOrganization(s));

  return (
    <div>
      <h1>Manage Sources</h1>
      <p className="muted" style={{ marginBottom: "18px" }}>
        The organizations and job boards checked for new postings every day, and suggestions for
        more. See the All Jobs page for what's turned up. For LinkedIn/Indeed/Glassdoor or anything
        you found yourself, use "Add a job" on the My Jobs page instead.
      </p>

      <section className="source-section">
        <h2 className="source-section-title">Organizations</h2>
        <p className="muted small">Single employers whose job boards you follow.</p>
        {organizations.map((s) => (
          <SourceCard key={s.id} source={s} onChanged={refresh} onFindMoreLikeThis={findMoreLikeThis} />
        ))}
        {organizations.length === 0 && (
          <p className="empty-state">No organizations yet — discover some below, or add one by URL.</p>
        )}
        <DiscoverCompaniesSection seed={orgSeed} onAdded={refresh} />
      </section>

      <section className="source-section">
        <h2 className="source-section-title">Job boards</h2>
        <p className="muted small">Multi-employer boards and job searches.</p>
        {boards.map((s) => (
          <SourceCard key={s.id} source={s} onChanged={refresh} onFindMoreLikeThis={findMoreLikeThis} />
        ))}
        {boards.length === 0 && (
          <p className="empty-state">No job boards yet — discover some below, or add one by URL.</p>
        )}
        <DiscoverJobBoardsSection seed={boardSeed} onAdded={refresh} />
      </section>

      <AddSourceForm onAdded={() => refresh()} />

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
