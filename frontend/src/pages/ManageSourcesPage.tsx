import { useEffect, useState } from "react";
import { api, RunProgress, runProgressText, Source, SourceKind } from "../api";
import { DiscoverCompaniesSection, DiscoverJobBoardsSection, DiscoverySeed } from "../components/Discovery";
import { Modal } from "../components/Modal";
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
      <h2>Add a source manually</h2>
      <p className="muted small">
        For anything "Add more" doesn't cover — like a keyword search on Adzuna or RemoteOK — or to
        set a source's details by hand. New postings feed into All Jobs and the daily digest.
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
        <p className="muted small field-hint">
          Your label for this source, shown here and on its jobs' cards.
        </p>
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
        <span className="flabel">Employer name (optional)</span>
        <input
          placeholder="e.g. 'Stripe'"
          value={companyName}
          onChange={(e) => setCompanyName(e.target.value)}
          style={{ width: "100%" }}
        />
        <p className="muted small field-hint">
          Fill this in for one employer's careers page: it's the company shown on jobs that don't
          name one, and it files the source under Organizations. Leave it blank for a job board or
          search that lists many employers.
        </p>
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

// Short labels for a source row; KIND_LABELS (the manual form's options) are longer.
const SHORT_KIND: Record<SourceKind, string> = {
  ats_greenhouse: "Greenhouse",
  ats_lever: "Lever",
  ats_ashby: "Ashby",
  aggregator_adzuna: "Adzuna search",
  aggregator_remoteok: "RemoteOK",
  custom_board: "Web page",
};

// Where a source reads from, in a line: the page's host+path, or the ATS board / search term.
function sourceWhere(source: Source): string {
  const c = source.config;
  if (typeof c.url === "string") return c.url.replace(/^https?:\/\/(www\.)?/, "").replace(/\/$/, "");
  const value = c.board_token ?? c.company_slug ?? c.board_name ?? c.what ?? c.tags;
  return Array.isArray(value) ? value.join(", ") : String(value ?? "");
}

// The page a person would open to see this source's jobs: the web page itself, or the ATS's public
// board. The keyword aggregators (Adzuna, RemoteOK) are searches, not one page, so they get none.
function sourceLink(source: Source): string | null {
  const c = source.config;
  const token = (key: string) => (typeof c[key] === "string" ? encodeURIComponent(c[key] as string) : null);
  switch (source.kind) {
    case "custom_board":
      return typeof c.url === "string" ? c.url : null;
    case "ats_greenhouse":
      return token("board_token") && `https://job-boards.greenhouse.io/${token("board_token")}`;
    case "ats_lever":
      return token("company_slug") && `https://jobs.lever.co/${token("company_slug")}`;
    case "ats_ashby":
      return token("board_name") && `https://jobs.ashbyhq.com/${token("board_name")}`;
    default:
      return null;
  }
}

// SQLite's datetime('now') is UTC without a zone marker.
function parseDbTime(value: string): number {
  return new Date(value.includes("T") ? value : `${value.replace(" ", "T")}Z`).getTime();
}

function timeAgo(value: string): string {
  const minutes = Math.round((Date.now() - parseDbTime(value)) / 60_000);
  if (minutes < 2) return "just now";
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

// A just-added source is checked in the background (daily_run.schedule_source_refresh); one that's
// still never been checked a while later is just waiting for the daily run.
const CHECKING_WINDOW_MS = 5 * 60_000;

function isChecking(source: Source): boolean {
  return !!source.enabled && !source.last_polled_at && Date.now() - parseDbTime(source.created_at) < CHECKING_WINDOW_MS;
}

function SourceStatus({ source }: { source: Source }) {
  if (!source.enabled) return <span className="pill">disabled</span>;
  if (isChecking(source)) return <Spinner label="Checking for jobs..." />;
  if (!source.last_polled_at) return <span className="muted">Not checked yet</span>;
  return (
    <span className="muted">
      {source.job_count} {source.job_count === 1 ? "job" : "jobs"} · checked {timeAgo(source.last_polled_at)}
    </span>
  );
}

function SourceRow({
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
      setTestResult(
        r.count ? `Found ${r.count} posting(s), e.g. "${r.preview[0]?.title}"` : "Found no postings on this page.",
      );
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

  const link = sourceLink(source);

  return (
    <li className={`source-row${source.enabled ? "" : " off"}`}>
      <div className="source-row-top">
        {link ? (
          <a className="source-row-name" href={link} target="_blank" rel="noreferrer" title={`Open ${link}`}>
            {source.name}
          </a>
        ) : (
          <span className="source-row-name">{source.name}</span>
        )}
        <span className="source-row-status small">
          <SourceStatus source={source} />
        </span>
      </div>
      <div className="source-row-where small" title={JSON.stringify(source.config)}>
        {SHORT_KIND[source.kind]} · {sourceWhere(source)}
      </div>
      {source.last_error && (
        <p className="error-text small source-row-error" title={source.last_error}>
          Last check failed: {source.last_error}
        </p>
      )}
      <div className="source-row-actions small">
        <button className="link small" onClick={test} disabled={busy}>
          {busy ? <Spinner label="Testing..." /> : "Test fetch"}
        </button>
        <button className="link small" onClick={() => onFindMoreLikeThis(source)}>
          More like this
        </button>
        <button className="link small" onClick={toggle}>
          {source.enabled ? "Disable" : "Enable"}
        </button>
        <button className="link small danger" onClick={remove}>
          Delete
        </button>
      </div>
      {testResult && <p className="muted small source-row-test">{testResult}</p>}
    </li>
  );
}

type SectionKind = "organizations" | "boards";
type DialogTab = "link" | "discover";

const DETECTED_LABELS: Partial<Record<SourceKind, string>> = {
  ats_greenhouse: "Greenhouse board — new postings come straight from its API",
  ats_lever: "Lever board — new postings come straight from its API",
  ats_ashby: "Ashby board — new postings come straight from its API",
  custom_board: "Careers page — checked like a web page each day",
};

function AddByLinkForm({ kind, onAdded }: { kind: SectionKind; onAdded: () => void }) {
  const [name, setName] = useState("");
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const detected = detectSourceFromUrl(url);
  const org = kind === "organizations";

  const submit = async () => {
    if (!name.trim() || !detected) return;
    setBusy(true);
    setError(null);
    try {
      await api.createSource({
        kind: detected.kind,
        name: name.trim(),
        // An organization name is what files it under Organizations (see isOrganization).
        company_name: org ? name.trim() : null,
        config: detected.config,
      });
      setName("");
      setUrl("");
      onAdded();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <label className="field">
        <span>Name</span>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder={org ? "e.g. Nava PBC" : "e.g. Tech Jobs for Good"}
        />
      </label>
      <label className="field">
        <span>{org ? "Link to their jobs page" : "Link to the board"}</span>
        <input
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder={org ? "e.g. https://boards.greenhouse.io/nava" : "e.g. https://techjobsforgood.com"}
        />
      </label>
      <p className="muted small" style={{ marginTop: 0 }}>
        {detected
          ? DETECTED_LABELS[detected.kind]
          : url.trim()
            ? "Enter a full link, starting with https://"
            : "Paste the page where their open jobs are listed."}
      </p>
      {error && <p className="error-text small">{error}</p>}
      <button className="primary" onClick={submit} disabled={busy || !name.trim() || !detected}>
        {busy ? <Spinner label="Adding..." /> : org ? "Add organization" : "Add job board"}
      </button>
    </div>
  );
}

// "Add more" for one section: add a source by link, or discover suggestions. Discovery stays
// mounted inside the (closed) dialog, so its results survive closing and reopening.
function AddMoreDialog({
  kind,
  open,
  tab,
  onTab,
  onClose,
  seed,
  onAdded,
}: {
  kind: SectionKind;
  open: boolean;
  tab: DialogTab;
  onTab: (tab: DialogTab) => void;
  onClose: () => void;
  seed: DiscoverySeed | null;
  onAdded: () => void;
}) {
  const org = kind === "organizations";
  return (
    <Modal open={open} onClose={onClose} title={org ? "Add organizations" : "Add job boards"}>
      <div className="seg" style={{ marginBottom: "16px" }}>
        <button className={tab === "link" ? "on" : ""} onClick={() => onTab("link")}>
          Add by link
        </button>
        <button className={tab === "discover" ? "on" : ""} onClick={() => onTab("discover")}>
          Discover suggestions
        </button>
      </div>
      <div hidden={tab !== "link"}>
        <AddByLinkForm
          kind={kind}
          onAdded={() => {
            onAdded();
            onClose();
          }}
        />
      </div>
      <div hidden={tab !== "discover"}>
        {org ? (
          <DiscoverCompaniesSection bare seed={seed} onAdded={onAdded} />
        ) : (
          <DiscoverJobBoardsSection bare seed={seed} onAdded={onAdded} />
        )}
      </div>
    </Modal>
  );
}

// One column of the page: a header with "Add more", then a fixed-height, scrolling list.
function SourcePanel({
  title,
  blurb,
  empty,
  sources,
  onAddMore,
  onChanged,
  onFindMoreLikeThis,
}: {
  title: string;
  blurb: string;
  empty: string;
  sources: Source[];
  onAddMore: () => void;
  onChanged: () => void;
  onFindMoreLikeThis: (source: Source) => void;
}) {
  return (
    <section className="card source-panel">
      <div className="source-section-head">
        <div>
          <h2 className="source-section-title">
            {title} <span className="muted source-count">{sources.length}</span>
          </h2>
          <p className="muted small">{blurb}</p>
        </div>
        <button className="primary" onClick={onAddMore}>
          + Add more
        </button>
      </div>
      {sources.length === 0 ? (
        <p className="empty-state">{empty}</p>
      ) : (
        <ul className="source-list">
          {sources.map((s) => (
            <SourceRow key={s.id} source={s} onChanged={onChanged} onFindMoreLikeThis={onFindMoreLikeThis} />
          ))}
        </ul>
      )}
    </section>
  );
}

export default function ManageSourcesPage() {
  const [sources, setSources] = useState<Source[]>([]);
  const [runResult, setRunResult] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [runProgress, setRunProgress] = useState<RunProgress | null>(null);
  const [orgSeed, setOrgSeed] = useState<DiscoverySeed | null>(null);
  const [boardSeed, setBoardSeed] = useState<DiscoverySeed | null>(null);
  const [dialog, setDialog] = useState<SectionKind | null>(null);
  const [dialogTab, setDialogTab] = useState<DialogTab>("link");

  const openDialog = (kind: SectionKind, tab: DialogTab = "link") => {
    setDialogTab(tab);
    setDialog(kind);
  };

  const refresh = () => api.listSources().then(setSources);

  // Opens the matching "Add more" dialog on its Discover tab and starts a "more like this" search.
  const findMoreLikeThis = (source: Source) => {
    const seed = { like: describeForDiscovery(source), label: source.name, nonce: Date.now() };
    const org = isOrganization(source);
    (org ? setOrgSeed : setBoardSeed)(seed);
    openDialog(org ? "organizations" : "boards", "discover");
  };

  useEffect(() => {
    refresh();
  }, []);

  // While a just-added source is being checked in the background, refresh the lists so its job
  // count appears when the check finishes.
  const checking = sources.some(isChecking);
  useEffect(() => {
    if (!checking) return;
    const timer = setInterval(refresh, 4000);
    return () => clearInterval(timer);
  }, [checking]);

  const runNow = async () => {
    setRunning(true);
    setRunResult(null);
    setRunProgress(null);
    try {
      const r = await api.runDailyNow(undefined, setRunProgress);
      const errors = r.errors ?? [];
      setRunResult(
        `Checked ${r.sources_polled ?? 0} sources, found ${r.jobs_fetched ?? 0} postings, scored ` +
          `${r.llm_scored_count ?? 0} new jobs.` +
          (errors.length ? `\n\n${errors.length} problem(s):\n${errors.join("\n")}` : ""),
      );
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

      <div className="sources-grid">
        <SourcePanel
          title="Organizations"
          blurb="Single employers whose job boards you follow."
          empty='No organizations yet — use "Add more" to find or add some.'
          sources={organizations}
          onAddMore={() => openDialog("organizations")}
          onChanged={refresh}
          onFindMoreLikeThis={findMoreLikeThis}
        />
        <SourcePanel
          title="Job boards"
          blurb="Multi-employer boards and job searches."
          empty='No job boards yet — use "Add more" to find or add some.'
          sources={boards}
          onAddMore={() => openDialog("boards")}
          onChanged={refresh}
          onFindMoreLikeThis={findMoreLikeThis}
        />
      </div>

      <AddMoreDialog
        kind="organizations"
        open={dialog === "organizations"}
        tab={dialogTab}
        onTab={setDialogTab}
        onClose={() => setDialog(null)}
        seed={orgSeed}
        onAdded={refresh}
      />
      <AddMoreDialog
        kind="boards"
        open={dialog === "boards"}
        tab={dialogTab}
        onTab={setDialogTab}
        onClose={() => setDialog(null)}
        seed={boardSeed}
        onAdded={refresh}
      />

      <AddSourceForm onAdded={() => refresh()} />

      <div className="card">
        <h2>Check every source now</h2>
        <p className="muted small">
          This runs automatically every day at 2:00pm ET: check every source, score the most promising new
          jobs, and update New &amp; Noteworthy. New sources are checked as soon as you add them.
        </p>
        <button className="ghost" onClick={runNow} disabled={running}>
          {running ? <Spinner label={runProgressText(runProgress, "Finishing up...")} /> : "Run now"}
        </button>
        {runResult && (
          <p className="small" style={{ whiteSpace: "pre-wrap", marginTop: "10px" }}>
            {runResult}
          </p>
        )}
      </div>
    </div>
  );
}
