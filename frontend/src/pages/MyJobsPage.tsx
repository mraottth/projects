import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, Application, APPLICATION_STATUS_LABELS, ApplicationStatus, Job } from "../api";
import { salaryText, salaryValue } from "../components/ScoredJobCard";
import { Spinner } from "../components/Spinner";

const STATUSES: ApplicationStatus[] = [
  "saved",
  "applied",
  "phone_screen",
  "interview",
  "offer",
  "rejected",
  "withdrawn",
];

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
      {STATUSES.map((s) => (
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
          className="ghost small"
          href={api.generatedDocumentDownloadUrl(app.resume_doc_id)}
          download
        >
          Download resume
        </a>
      ) : (
        <button className="ghost small" onClick={() => generate("resume")} disabled={busy !== null}>
          {busy === "resume" ? <Spinner label="Tailoring..." /> : "Tailor my resume"}
        </button>
      )}
      {app.cover_letter_doc_id ? (
        <a
          className="ghost small"
          href={api.generatedDocumentDownloadUrl(app.cover_letter_doc_id)}
          download
        >
          Download cover letter
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

type SortKey = "company" | "title" | "location" | "salary" | "applied_at" | "status";
type SortDir = "asc" | "desc";

function sortValue(app: Application, key: SortKey): string | number {
  switch (key) {
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
      return app.status;
  }
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

export default function MyJobsPage() {
  const [apps, setApps] = useState<Application[]>([]);
  const [pending, setPending] = useState<Job[]>([]);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<ApplicationStatus | "all">("all");
  const [sortKey, setSortKey] = useState<SortKey>("applied_at");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const refresh = () => api.listApplications().then(setApps);

  useEffect(() => {
    refresh();
  }, []);

  const updateInPlace = (updated: Application) =>
    setApps((prev) => prev.map((a) => (a.id === updated.id ? updated : a)));

  const onSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("asc");
    }
  };

  const query = search.trim().toLowerCase();
  const filtered = apps.filter((app) => {
    if (statusFilter !== "all" && app.status !== statusFilter) return false;
    if (query && !`${app.company} ${app.title} ${app.location ?? ""}`.toLowerCase().includes(query)) {
      return false;
    }
    return true;
  });

  const sorted = [...filtered].sort((a, b) => {
    const av = sortValue(a, sortKey);
    const bv = sortValue(b, sortKey);
    const cmp = av < bv ? -1 : av > bv ? 1 : 0;
    return sortDir === "asc" ? cmp : -cmp;
  });

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

      {apps.length > 0 && (
        <div className="card">
          <div className="form-row" style={{ marginBottom: 0 }}>
            <div className="field" style={{ marginBottom: 0, flex: "1 1 260px" }}>
              <span>Search</span>
              <input
                placeholder="Search company, title, location..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
            </div>
            <div className="field" style={{ marginBottom: 0 }}>
              <span>Status</span>
              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value as ApplicationStatus | "all")}
              >
                <option value="all">All</option>
                {STATUSES.map((s) => (
                  <option key={s} value={s}>
                    {APPLICATION_STATUS_LABELS[s]}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>
      )}

      {apps.length > 0 && sorted.length === 0 && (
        <p className="empty-state">No jobs match these filters.</p>
      )}

      {sorted.length > 0 && (
        <div className="card" style={{ overflowX: "auto" }}>
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
