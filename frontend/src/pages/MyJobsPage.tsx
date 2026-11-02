import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  api,
  Application,
  ApplicationInsights,
  APPLICATION_STATUS_LABELS,
  APPLICATION_STATUSES,
  ApplicationStatus,
  Job,
} from "../api";
import { Markdown } from "../components/Markdown";
import { FitBadge, salaryText, salaryValue } from "../components/ScoredJobCard";
import { Spinner } from "../components/Spinner";

function AddJobForm({ onAdded }: { onAdded: (job: Job) => void }) {
  const [mode, setMode] = useState<"url" | "text">("url");
  const [value, setValue] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    if (!value.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const job = await api.createManualJob(mode === "url" ? { url: value } : { raw_text: value });
      setValue("");
      onAdded(job);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card">
      <h2 style={{ marginBottom: "8px" }}>Add a job</h2>
      {mode === "url" ? (
        <div className="form-row" style={{ marginBottom: "4px" }}>
          <input
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder="Paste a job URL..."
            style={{ flex: "1 1 320px" }}
          />
          <button className="primary" onClick={submit} disabled={busy}>
            {busy ? <Spinner label="Adding..." /> : "Add"}
          </button>
        </div>
      ) : (
        <>
          <textarea
            rows={4}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder="Paste the job description here..."
            style={{ width: "100%", display: "block", marginBottom: "6px" }}
          />
          <div className="form-actions" style={{ marginTop: 0, marginBottom: "4px" }}>
            <button className="primary" onClick={submit} disabled={busy}>
              {busy ? <Spinner label="Adding..." /> : "Add"}
            </button>
          </div>
        </>
      )}
      <button className="link small" onClick={() => setMode(mode === "url" ? "text" : "url")}>
        {mode === "url" ? "Paste the full text instead" : "Use a URL instead"}
      </button>
      {error && <p className="error-text small">{error}</p>}
    </div>
  );
}

function PendingReview({ job, onConfirmed }: { job: Job; onConfirmed: (job: Job) => void }) {
  const [draft, setDraft] = useState(job);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const confirm = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.updateJob(draft.id, draft);
      const confirmed = await api.confirmJob(draft.id);
      onConfirmed(confirmed);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card">
      <h3>Review before scoring</h3>
      <div className="field">
        <span>Title</span>
        <input value={draft.title} onChange={(e) => setDraft({ ...draft, title: e.target.value })} />
      </div>
      <div className="field">
        <span>Company</span>
        <input value={draft.company} onChange={(e) => setDraft({ ...draft, company: e.target.value })} />
      </div>
      <div className="field">
        <span>Location</span>
        <input
          value={draft.location ?? ""}
          onChange={(e) => setDraft({ ...draft, location: e.target.value })}
          placeholder="Location"
        />
      </div>
      <button className="primary" onClick={confirm} disabled={busy}>
        {busy ? <Spinner label="Scoring..." /> : "Confirm & add to My Jobs"}
      </button>
      {error && <p className="error-text small">{error}</p>}
    </div>
  );
}

function StatusSelect({ app, onChanged }: { app: Application; onChanged: (app: Application) => void }) {
  const [busy, setBusy] = useState(false);

  const change = async (status: ApplicationStatus) => {
    setBusy(true);
    try {
      const updated = await api.updateApplication(app.id, { status });
      onChanged(updated);
    } finally {
      setBusy(false);
    }
  };

  return (
    <select
      className={`status-select pill-${app.status}`}
      value={app.status}
      disabled={busy}
      onChange={(e) => change(e.target.value as ApplicationStatus)}
    >
      {APPLICATION_STATUSES.map((s) => (
        <option key={s} value={s}>
          {APPLICATION_STATUS_LABELS[s]}
        </option>
      ))}
    </select>
  );
}

function AppliedDateCell({ app, onChanged }: { app: Application; onChanged: (app: Application) => void }) {
  const [busy, setBusy] = useState(false);
  const dateValue = app.applied_at ? app.applied_at.slice(0, 10) : "";

  const change = async (value: string) => {
    if (!value) return;
    setBusy(true);
    try {
      const updated = await api.updateApplication(app.id, { applied_at: value });
      onChanged(updated);
    } finally {
      setBusy(false);
    }
  };

  return (
    <input
      type="date"
      value={dateValue}
      disabled={busy}
      onChange={(e) => change(e.target.value)}
      style={{ fontSize: "0.85rem", padding: "4px 6px", width: "140px" }}
    />
  );
}

function DocActions({ app, onChanged }: { app: Application; onChanged: (app: Application) => void }) {
  const [busy, setBusy] = useState<"resume" | "cover-letter" | null>(null);
  const [error, setError] = useState<string | null>(null);

  const generate = async (kind: "resume" | "cover-letter") => {
    setBusy(kind);
    setError(null);
    try {
      if (kind === "resume") {
        const result = await api.generateResume(app.job_id, app.id);
        onChanged({ ...app, resume_doc_id: result.id });
      } else {
        const result = await api.generateCoverLetter(app.job_id, app.id);
        onChanged({ ...app, cover_letter_doc_id: result.id });
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="doc-actions">
      {app.resume_doc_id ? (
        <a
          className="ghost small doc-download"
          href={api.generatedDocumentDownloadUrl(app.resume_doc_id)}
          download
        >
          <span aria-hidden="true">↓</span> Download resume
        </a>
      ) : (
        <button className="ghost small" onClick={() => generate("resume")} disabled={busy !== null}>
          {busy === "resume" ? <Spinner label="Tailoring..." /> : "Tailor my resume"}
        </button>
      )}
      {app.cover_letter_doc_id ? (
        <a
          className="ghost small doc-download"
          href={api.generatedDocumentDownloadUrl(app.cover_letter_doc_id)}
          download
        >
          <span aria-hidden="true">↓</span> Download cover letter
        </a>
      ) : (
        <button className="ghost small" onClick={() => generate("cover-letter")} disabled={busy !== null}>
          {busy === "cover-letter" ? <Spinner label="Tailoring..." /> : "Tailor my cover letter"}
        </button>
      )}
      <Link className="ghost small jobby-link" to={`/jobby?job=${app.job_id}`}>
        Prep with Jobby
      </Link>
      {error && <span className="error-text small">{error}</span>}
    </div>
  );
}

function RemoveButton({ app, onRemoved }: { app: Application; onRemoved: (id: number) => void }) {
  const [busy, setBusy] = useState(false);

  const remove = async () => {
    if (!window.confirm(`Remove ${app.title} at ${app.company} from My Jobs?`)) return;
    setBusy(true);
    try {
      await api.deleteApplication(app.id);
      onRemoved(app.id);
    } finally {
      setBusy(false);
    }
  };

  return (
    <button
      className="link small danger remove-x"
      onClick={remove}
      disabled={busy}
      aria-label="Remove from My Jobs"
      title="Remove from My Jobs"
    >
      {busy ? <Spinner /> : "×"}
    </button>
  );
}

type SortKey = "score" | "company" | "title" | "location" | "salary" | "applied_at" | "status";
type SortDir = "asc" | "desc";

function scoreOf(app: Application): number {
  return (app.interest_score ?? -1) + (app.candidate_fit_score ?? -1);
}

function sortValue(app: Application, key: SortKey): string | number {
  switch (key) {
    case "score":
      return scoreOf(app);
    case "company":
      return app.company.toLowerCase();
    case "title":
      return app.title.toLowerCase();
    case "location":
      return (app.location ?? "").toLowerCase();
    case "salary":
      return salaryValue(app);
    case "applied_at":
      return app.applied_at ?? "";
    case "status":
      return APPLICATION_STATUSES.indexOf(app.status);
  }
}

// Ids in display order. "Best match" is the default: interest + fit score, then most recently
// applied; every other column breaks ties the same way.
function sortedIds(apps: Application[], key: SortKey, dir: SortDir): number[] {
  const tiebreak = (a: Application, b: Application) =>
    scoreOf(b) - scoreOf(a) || (b.applied_at ?? "").localeCompare(a.applied_at ?? "");
  return [...apps]
    .sort((a, b) => {
      const av = sortValue(a, key);
      const bv = sortValue(b, key);
      const cmp = av < bv ? -1 : av > bv ? 1 : 0;
      return (dir === "asc" ? cmp : -cmp) || tiebreak(a, b);
    })
    .map((a) => a.id);
}

function SortHeader({
  label,
  columnKey,
  sortKey,
  sortDir,
  onSort,
}: {
  label: string;
  columnKey: SortKey;
  sortKey: SortKey;
  sortDir: SortDir;
  onSort: (key: SortKey) => void;
}) {
  const active = sortKey === columnKey;
  return (
    <th style={{ cursor: "pointer", whiteSpace: "nowrap" }} onClick={() => onSort(columnKey)}>
      {label} {active && (sortDir === "asc" ? "▲" : "▼")}
    </th>
  );
}

const REJECTED: ApplicationStatus[] = ["rejected_no_interview", "rejected_interviewed"];
// Reached an interview at some point: what "getting a response" means here.
const INTERVIEWED: ApplicationStatus[] = ["interviewing", "offer", "rejected_interviewed"];

function pct(part: number, whole: number): string {
  return whole ? `${Math.round((100 * part) / whole)}%` : "–";
}

/** Counts from the loaded list (no API call), then Claude's read on what's working, on request. */
function Insights({ apps }: { apps: Application[] }) {
  const [analysis, setAnalysis] = useState<ApplicationInsights | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getApplicationInsights().then(setAnalysis).catch(() => {});
  }, []);

  const sent = apps.filter((a) => a.status !== "saved");
  const count = (statuses: ApplicationStatus[]) => sent.filter((a) => statuses.includes(a.status)).length;
  const interviewed = count(INTERVIEWED);
  const byFit = (["High", "Med", "Low"] as const)
    .map((fit) => {
      const group = sent.filter((a) => a.candidate_fit === fit);
      return { fit, sent: group.length, interviewed: group.filter((a) => INTERVIEWED.includes(a.status)).length };
    })
    .filter((g) => g.sent > 0);

  const build = async () => {
    setBusy(true);
    setError(null);
    try {
      setAnalysis(await api.buildApplicationInsights());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const stale = analysis !== null && analysis.applications !== apps.length;

  return (
    <div className="card insights">
      <h2>How your applications are going</h2>
      <div className="insight-stats">
        <div><strong>{sent.length}</strong><span>applied</span></div>
        <div><strong>{count(["waiting"])}</strong><span>waiting</span></div>
        <div><strong>{interviewed}</strong><span>got an interview ({pct(interviewed, sent.length)})</span></div>
        <div><strong>{count(["offer"])}</strong><span>offers</span></div>
        <div>
          <strong>{count(REJECTED)}</strong>
          <span>
            rejected ({count(["rejected_no_interview"])} without an interview)
          </span>
        </div>
      </div>
      {byFit.length > 0 && (
        <p className="small muted insight-fit">
          Interview rate by fit rating:{" "}
          {byFit.map((g, i) => (
            <span key={g.fit}>
              {i > 0 && " · "}
              {g.fit} fit {pct(g.interviewed, g.sent)} ({g.interviewed} of {g.sent})
            </span>
          ))}
        </p>
      )}
      <div className="insight-analysis">
        {analysis ? (
          <>
            <Markdown text={analysis.text} />
            <p className="small muted">
              Based on {analysis.applications} jobs, {new Date(analysis.generated_at).toLocaleDateString()}.
              {stale && " You've changed your list since then."}
            </p>
          </>
        ) : (
          <p className="small muted">
            Ask Claude which kinds of jobs are getting interviews and which aren't, from everything above.
          </p>
        )}
        <button className="ghost small" onClick={build} disabled={busy || sent.length === 0}>
          {busy ? <Spinner label="Reading your applications..." /> : analysis ? "Update the analysis" : "What's working?"}
        </button>
        {error && <p className="error-text small">{error}</p>}
      </div>
    </div>
  );
}

export default function MyJobsPage() {
  const [apps, setApps] = useState<Application[]>([]);
  const [pending, setPending] = useState<Job[]>([]);
  const [search, setSearch] = useState("");
  const [shownStatuses, setShownStatuses] = useState<ApplicationStatus[]>(APPLICATION_STATUSES);
  const [sortKey, setSortKey] = useState<SortKey>("score");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  // The row order is fixed when the list loads or a sort is chosen, not on every edit: re-sorting as
  // soon as an applied date changed moved the row out from under the cursor (and its date picker).
  const [order, setOrder] = useState<number[]>([]);
  const sortRef = useRef({ key: sortKey, dir: sortDir });

  const refresh = () =>
    api.listApplications().then((list) => {
      setApps(list);
      setOrder(sortedIds(list, sortRef.current.key, sortRef.current.dir));
    });

  useEffect(() => {
    refresh();
  }, []);

  const updateInPlace = (updated: Application) =>
    setApps((prev) => prev.map((a) => (a.id === updated.id ? updated : a)));

  const applySort = (key: SortKey, dir: SortDir) => {
    setSortKey(key);
    setSortDir(dir);
    sortRef.current = { key, dir };
    setOrder(sortedIds(apps, key, dir));
  };

  const onSort = (key: SortKey) => {
    if (sortKey === key) applySort(key, sortDir === "asc" ? "desc" : "asc");
    else applySort(key, key === "applied_at" || key === "score" || key === "salary" ? "desc" : "asc");
  };

  const toggleStatus = (status: ApplicationStatus) =>
    setShownStatuses((prev) => (prev.includes(status) ? prev.filter((s) => s !== status) : [...prev, status]));

  const query = search.trim().toLowerCase();
  const matches = (app: Application) =>
    shownStatuses.includes(app.status) &&
    (!query || `${app.company} ${app.title} ${app.location ?? ""}`.toLowerCase().includes(query));

  const byId = new Map(apps.map((a) => [a.id, a]));
  const ordered = [
    ...order.map((id) => byId.get(id)).filter((a): a is Application => a !== undefined),
    ...apps.filter((a) => !order.includes(a.id)),
  ];
  const sorted = ordered.filter(matches);
  const outOfOrder = sortedIds(apps, sortKey, sortDir).join() !== ordered.map((a) => a.id).join();
  const countByStatus = (status: ApplicationStatus) => apps.filter((a) => a.status === status).length;

  return (
    <div>
      <h1>My Jobs</h1>
      <p className="muted" style={{ marginBottom: "18px" }}>
        Every job you've marked interested or added directly — status tracked from drafting
        through offer.
      </p>

      <AddJobForm onAdded={(job) => setPending((p) => [...p, job])} />

      {pending.map((job) => (
        <PendingReview
          key={job.id}
          job={job}
          onConfirmed={() => {
            setPending((p) => p.filter((j) => j.id !== job.id));
            refresh();
          }}
        />
      ))}

      {apps.length === 0 && (
        <p className="empty-state">
          No jobs tracked yet — mark one "Interested" on New &amp; Noteworthy or All Jobs, or add one above.
        </p>
      )}

      {apps.length > 0 && <Insights apps={apps} />}

      {apps.length > 0 && (
        <div className="card">
          <div className="field" style={{ marginBottom: "10px" }}>
            <span>Search</span>
            <input
              placeholder="Search company, title, location..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
          <div className="status-filter">
            <span className="flabel">Show</span>
            <div className="chip-toggle-group">
              {APPLICATION_STATUSES.map((s) => (
                <button
                  key={s}
                  type="button"
                  className={`chip-toggle${shownStatuses.includes(s) ? " on" : ""}`}
                  onClick={() => toggleStatus(s)}
                >
                  {APPLICATION_STATUS_LABELS[s]} <span className="chip-count">{countByStatus(s)}</span>
                </button>
              ))}
            </div>
            {shownStatuses.length < APPLICATION_STATUSES.length && (
              <button className="link small" onClick={() => setShownStatuses(APPLICATION_STATUSES)}>
                Show all
              </button>
            )}
          </div>
        </div>
      )}

      {apps.length > 0 && sorted.length === 0 && (
        <p className="empty-state">No jobs match these filters.</p>
      )}

      {sorted.length > 0 && (
        <div className="card" style={{ overflowX: "auto" }}>
          <div className="table-tools small">
            {sortKey !== "score" && (
              <button className="link small" onClick={() => applySort("score", "desc")}>
                Sort by best match
              </button>
            )}
            {outOfOrder && (
              <button className="link small" onClick={() => applySort(sortKey, sortDir)}>
                Re-sort with your changes
              </button>
            )}
          </div>
          <table className="jobs-table">
            <thead>
              <tr>
                <SortHeader label="Company" columnKey="company" sortKey={sortKey} sortDir={sortDir} onSort={onSort} />
                <SortHeader label="Title" columnKey="title" sortKey={sortKey} sortDir={sortDir} onSort={onSort} />
                <SortHeader
                  label="Location"
                  columnKey="location"
                  sortKey={sortKey}
                  sortDir={sortDir}
                  onSort={onSort}
                />
                <SortHeader label="Salary" columnKey="salary" sortKey={sortKey} sortDir={sortDir} onSort={onSort} />
                <SortHeader
                  label="Applied Date"
                  columnKey="applied_at"
                  sortKey={sortKey}
                  sortDir={sortDir}
                  onSort={onSort}
                />
                <SortHeader label="Status" columnKey="status" sortKey={sortKey} sortDir={sortDir} onSort={onSort} />
                <th>Documents</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((app) => (
                <tr key={app.id}>
                  <td>{app.company}</td>
                  <td>
                    <Link to={`/jobs/${app.job_id}`}>{app.title}</Link>
                    {(app.interest_fit || app.candidate_fit) && (
                      <div className="title-badges">
                        <FitBadge label="Interest" value={app.interest_fit} />
                        <FitBadge label="Fit" value={app.candidate_fit} />
                      </div>
                    )}
                  </td>
                  <td>{app.location ?? "N/A"}</td>
                  <td>{salaryText(app)}</td>
                  <td>
                    <AppliedDateCell app={app} onChanged={updateInPlace} />
                  </td>
                  <td>
                    <StatusSelect app={app} onChanged={updateInPlace} />
                  </td>
                  <td>
                    <DocActions app={app} onChanged={updateInPlace} />
                  </td>
                  <td>
                    <RemoveButton
                      app={app}
                      onRemoved={(id) => setApps((prev) => prev.filter((a) => a.id !== id))}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
