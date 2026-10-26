import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, Application, APPLICATION_STATUS_LABELS, ApplicationStatus, Job } from "../api";
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
  const [mode, setMode] = useState<"url" | "text">("text");
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
      <h2>Add a job</h2>
      <p className="muted small">
        Found something yourself — on LinkedIn, Indeed, Glassdoor, anywhere? Paste the URL or the
        raw job text (if the URL fetch fails — common on sites that require login — paste the text
        instead). It gets scored and added straight to My Jobs below.
      </p>
      <div className="seg" style={{ marginBottom: "10px" }}>
        <button type="button" className={mode === "text" ? "on" : ""} onClick={() => setMode("text")}>
          Raw text
        </button>
        <button type="button" className={mode === "url" ? "on" : ""} onClick={() => setMode("url")}>
          URL
        </button>
      </div>
      {mode === "text" ? (
        <textarea
          rows={6}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Paste the job description here..."
          style={{ width: "100%", display: "block" }}
        />
      ) : (
        <input
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="https://..."
          style={{ width: "100%", display: "block" }}
        />
      )}
      <div className="form-actions">
        <button className="primary" onClick={submit} disabled={busy}>
          {busy ? <Spinner label="Parsing..." /> : "Parse"}
        </button>
      </div>
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

function StatusSelect({
  app,
  onChanged,
}: {
  app: Application;
  onChanged: (app: Application) => void;
}) {
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
    <select value={app.status} disabled={busy} onChange={(e) => change(e.target.value as ApplicationStatus)}>
      {STATUSES.map((s) => (
        <option key={s} value={s}>
          {APPLICATION_STATUS_LABELS[s]}
        </option>
      ))}
    </select>
  );
}

export default function MyJobsPage() {
  const [apps, setApps] = useState<Application[]>([]);
  const [pending, setPending] = useState<Job[]>([]);

  const refresh = () => api.listApplications().then(setApps);

  useEffect(() => {
    refresh();
  }, []);

  const updateInPlace = (updated: Application) =>
    setApps((prev) => prev.map((a) => (a.id === updated.id ? updated : a)));

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
          No jobs tracked yet — mark one "Interested" on the Digest or My Sources, or add one above.
        </p>
      )}

      {apps.map((app) => (
        <div className="card" key={app.id}>
          <div className="card-header">
            <div>
              <div className="card-title">
                <Link to={`/jobs/${app.job_id}`}>
                  {app.title} — {app.company}
                </Link>
              </div>
              <div className="card-meta">{app.location ?? "location unknown"}</div>
            </div>
            <span className={`pill pill-${app.status}`}>{APPLICATION_STATUS_LABELS[app.status]}</span>
          </div>

          <div className="form-row">
            <div className="field" style={{ marginBottom: 0 }}>
              <span>Status</span>
              <StatusSelect app={app} onChanged={updateInPlace} />
            </div>
          </div>

          <div className="card-section small muted">
            <strong>History: </strong>
            {app.history
              .map((h) => APPLICATION_STATUS_LABELS[h.status as ApplicationStatus] ?? h.status)
              .join(" → ")}
          </div>
        </div>
      ))}
    </div>
  );
}
