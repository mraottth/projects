import type React from "react";
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
import { JobbyLogo } from "../components/JobbyLogo";
import { Markdown } from "../components/Markdown";
import { Modal } from "../components/Modal";
import { FitBadge, salaryText, salaryValue } from "../components/ScoredJobCard";
import { RescoreButton } from "../components/RescoreButton";
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
    <div className="doc-actions stacked">
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

/** The Prep column: one small toggle that opens the tailoring and Jobby actions, stacked, inside
 * the cell. Showing all three inline made the column the widest in the table. */
function PrepCell({ app, onChanged }: { app: Application; onChanged: (app: Application) => void }) {
  const [open, setOpen] = useState(false);
  const ready = (app.resume_doc_id ? 1 : 0) + (app.cover_letter_doc_id ? 1 : 0);
  return (
    <div className="prep-cell">
      <button
        className={`prep-toggle${ready ? " has-ready" : ""}`}
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-label={open ? "Hide prep actions" : "Show prep actions"}
        title={
          (open ? "Hide" : "Tailor documents or prep with Jobby") +
          (ready ? ` (${ready} tailored document${ready > 1 ? "s" : ""} ready)` : "")
        }
      >
        {open ? "−" : "+"}
      </button>
      {open && <DocActions app={app} onChanged={onChanged} />}
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
// Still in play vs. over, for the table's Active / Closed toggle.
const CLOSED: ApplicationStatus[] = [...REJECTED, "withdrawn"];
type StatusView = "active" | "closed" | "all";

function inView(app: Application, view: StatusView): boolean {
  if (view === "all") return true;
  return CLOSED.includes(app.status) === (view === "closed");
}

// SQLite's datetime('now') is UTC without a zone marker.
function parseDbTime(value: string): number {
  return new Date(value.includes("T") ? value : `${value.replace(" ", "T")}Z`).getTime();
}

function pct(part: number, whole: number): string {
  return whole ? `${Math.round((100 * part) / whole)}%` : "–";
}

/** The analysis's opening sentence (the prompt asks for one before the first heading), or null. */
function takeaway(text: string): string | null {
  const intro = text.split(/\n#/)[0].trim();
  return intro && !intro.startsWith("#") ? intro : null;
}

/** Counts from the loaded list (no API call), then Jobby's read on what's working, on request. */
function Insights({ apps }: { apps: Application[] }) {
  const [analysis, setAnalysis] = useState<ApplicationInsights | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);

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

  const ask = async () => {
    setBusy(true);
    setError(null);
    try {
      setAnalysis(await api.buildApplicationInsights());
      setOpen(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  // Out of date once any application changed after Jobby wrote this (a status or date edit, a new
  // job), or one was removed (the count differs).
  const generatedAt = analysis ? new Date(analysis.generated_at).getTime() : 0;
  const stale =
    analysis !== null &&
    (analysis.applications !== apps.length || apps.some((a) => parseDbTime(a.updated_at) > generatedAt));
  const preview = analysis ? takeaway(analysis.text) : null;

  return (
    <div className="card insights">
      <h2>How your applications are going</h2>
      <div className="insight-stats">
        <div><strong>{sent.length}</strong><span>applied</span></div>
        <div><strong>{count(["waiting"])}</strong><span>waiting</span></div>
        <div><strong>{interviewed}</strong><span>interviews ({pct(interviewed, sent.length)})</span></div>
        <div><strong>{count(["offer"])}</strong><span>offers</span></div>
        <div>
          <strong>{count(REJECTED)}</strong>
          <span>rejected ({count(["rejected_no_interview"])} before interview)</span>
        </div>
      </div>
      {byFit.length > 0 && (
        <p className="small muted insight-fit">
          Interview rate by fit:{" "}
          {byFit.map((g, i) => (
            <span key={g.fit}>
              {i > 0 && " · "}
              {g.fit} {pct(g.interviewed, g.sent)} ({g.interviewed}/{g.sent})
            </span>
          ))}
        </p>
      )}
      <div className="jobby-take">
        <JobbyLogo size={40} />
        <div className="jobby-take-body">
          {analysis ? (
            <>
              {preview && <p className="jobby-take-preview">{preview}</p>}
              <div className="jobby-take-actions">
                <button className="ghost small" onClick={() => setOpen(true)}>
                  Read Jobby's take
                </button>
              </div>
            </>
          ) : (
            <>
              <p className="small muted">Jobby can read your applications and tell you what's landing and what isn't.</p>
              <button className="ghost small" onClick={ask} disabled={busy || sent.length === 0}>
                {busy ? <Spinner label="Jobby is reading..." /> : "Ask Jobby what's working"}
              </button>
            </>
          )}
          {error && <p className="error-text small">{error}</p>}
        </div>
      </div>
      <Modal
        open={open}
        onClose={() => setOpen(false)}
        title={
          <span className="jobby-modal-title">
            <JobbyLogo size={32} /> Jobby on your applications
          </span>
        }
      >
        {analysis && (
          <>
            <div className="jobby-updated small">
              <span className="muted">
                Last updated{" "}
                {new Date(analysis.generated_at).toLocaleString(undefined, {
                  month: "short",
                  day: "numeric",
                  hour: "numeric",
                  minute: "2-digit",
                })}
                , from {analysis.applications} jobs.
                {stale && " Your applications have changed since then."}
              </span>
              {stale && (
                <button className="ghost small" onClick={ask} disabled={busy}>
                  {busy ? <Spinner label="Jobby is re-reading..." /> : "↻ Refresh"}
                </button>
              )}
            </div>
            <Markdown text={analysis.text} />
            {error && <p className="error-text small">{error}</p>}
            <div className="form-actions">
              <Link className="ghost button-link" to="/jobby">
                Talk it over with Jobby
              </Link>
            </div>
          </>
        )}
      </Modal>
    </div>
  );
}

/** Keeps the toolbar and header row stuck under the app's top bar while the table scrolls with the
 * page: measures the top bar and toolbar and hands their heights to CSS. */
function useStickyOffsets(toolbar: React.RefObject<HTMLDivElement | null>, card: React.RefObject<HTMLDivElement | null>) {
  useEffect(() => {
    const topbar = document.querySelector<HTMLElement>(".topbar");
    const update = () => {
      if (!card.current || !toolbar.current) return;
      const top = topbar?.offsetHeight ?? 0;
      card.current.style.setProperty("--sticky-top", `${top}px`);
      card.current.style.setProperty("--thead-top", `${top + toolbar.current.offsetHeight}px`);
    };
    update();
    const observer = new ResizeObserver(update);
    if (topbar) observer.observe(topbar);
    if (toolbar.current) observer.observe(toolbar.current);
    return () => observer.disconnect();
  });
}

export default function MyJobsPage() {
  const [apps, setApps] = useState<Application[]>([]);
  const [pending, setPending] = useState<Job[]>([]);
  const [search, setSearch] = useState("");
  const [view, setView] = useState<StatusView>("active");
  const [sortKey, setSortKey] = useState<SortKey>("score");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  // The row order is fixed when the list loads or a sort is chosen, not on every edit: re-sorting as
  // soon as an applied date changed moved the row out from under the cursor (and its date picker).
  const [order, setOrder] = useState<number[]>([]);
  const sortRef = useRef({ key: sortKey, dir: sortDir });
  const toolbarRef = useRef<HTMLDivElement>(null);
  const tableCardRef = useRef<HTMLDivElement>(null);
  useStickyOffsets(toolbarRef, tableCardRef);

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

  const query = search.trim().toLowerCase();
  const matches = (app: Application) =>
    inView(app, view) &&
    (!query || `${app.company} ${app.title} ${app.location ?? ""}`.toLowerCase().includes(query));

  const byId = new Map(apps.map((a) => [a.id, a]));
  const ordered = [
    ...order.map((id) => byId.get(id)).filter((a): a is Application => a !== undefined),
    ...apps.filter((a) => !order.includes(a.id)),
  ];
  const sorted = ordered.filter(matches);
  const outOfOrder = sortedIds(apps, sortKey, sortDir).join() !== ordered.map((a) => a.id).join();
  const viewCount = (v: StatusView) => apps.filter((a) => inView(a, v)).length;
  const VIEWS: { key: StatusView; label: string }[] = [
    { key: "active", label: "Active" },
    { key: "closed", label: "Closed" },
    { key: "all", label: "All" },
  ];

  return (
    <div>
      <h1>My Jobs</h1>
      <p className="muted" style={{ marginBottom: "18px" }}>
        Every job you've marked interested or added directly — status tracked from drafting
        through offer.
      </p>
      <RescoreButton scope="my_jobs" onDone={refresh} />

      <div className={`myjobs-top${apps.length > 0 ? "" : " single"}`}>
        {apps.length > 0 && <Insights apps={apps} />}
        <AddJobForm onAdded={(job) => setPending((p) => [...p, job])} />
      </div>

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
        <div className="card myjobs-table-card" ref={tableCardRef}>
          <div className="myjobs-toolbar" ref={toolbarRef}>
            <input
              className="myjobs-search"
              placeholder="Search company, title, location..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              aria-label="Search"
            />
            <div className="seg" role="group" aria-label="Which jobs to show">
              {VIEWS.map((v) => (
                <button
                  key={v.key}
                  className={view === v.key ? "on" : ""}
                  onClick={() => setView(v.key)}
                  title={
                    v.key === "active"
                      ? "Drafting, waiting, interviewing and offers"
                      : v.key === "closed"
                        ? "Rejected and withdrawn"
                        : undefined
                  }
                >
                  {v.label} <span className="chip-count">{viewCount(v.key)}</span>
                </button>
              ))}
            </div>
            <div className="table-tools small">
              <a className="link small" href={api.applicationsCsvUrl} download>
                Export CSV
              </a>
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
          </div>

          {sorted.length === 0 ? (
            <p className="empty-state">No jobs match.</p>
          ) : (
            <table className="jobs-table">
              <thead>
                <tr>
                  <SortHeader label="Title" columnKey="title" sortKey={sortKey} sortDir={sortDir} onSort={onSort} />
                  <SortHeader label="Organization" columnKey="company" sortKey={sortKey} sortDir={sortDir} onSort={onSort} />
                  <SortHeader label="Location" columnKey="location" sortKey={sortKey} sortDir={sortDir} onSort={onSort} />
                  <SortHeader label="Salary" columnKey="salary" sortKey={sortKey} sortDir={sortDir} onSort={onSort} />
                  <SortHeader label="Applied" columnKey="applied_at" sortKey={sortKey} sortDir={sortDir} onSort={onSort} />
                  <SortHeader label="Status" columnKey="status" sortKey={sortKey} sortDir={sortDir} onSort={onSort} />
                  <th>Prep</th>
                  <th aria-label="Remove"></th>
                </tr>
              </thead>
              <tbody>
                {sorted.map((app) => (
                  <tr key={app.id}>
                    <td className="col-title">
                      <Link to={`/jobs/${app.job_id}`}>{app.title}</Link>
                      {(app.interest_fit || app.candidate_fit) && (
                        <div className="title-badges">
                          <FitBadge label="Interest" value={app.interest_fit} />
                          <FitBadge label="Fit" value={app.candidate_fit} />
                        </div>
                      )}
                    </td>
                    <td className="col-company">{app.company}</td>
                    <td className="col-location">{app.location ?? "N/A"}</td>
                    <td className="col-nowrap">{salaryText(app)}</td>
                    <td className="col-nowrap">
                      <AppliedDateCell app={app} onChanged={updateInPlace} />
                    </td>
                    <td className="col-nowrap">
                      <StatusSelect app={app} onChanged={updateInPlace} />
                    </td>
                    <td className="col-nowrap">
                      <PrepCell app={app} onChanged={updateInPlace} />
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
          )}
        </div>
      )}
    </div>
  );
}
