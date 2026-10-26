import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, DigestEntry, Job } from "../api";
import { Markdown } from "../components/Markdown";
import { Spinner } from "../components/Spinner";

function daysAgo(iso: string): string {
  const days = Math.floor((Date.now() - new Date(iso).getTime()) / 86_400_000);
  if (days <= 0) return "today";
  if (days === 1) return "1 day ago";
  return `${days} days ago`;
}

function salaryText(job: Job): string | null {
  if (!job.salary_min && !job.salary_max) return null;
  const fmt = (n: number) => `$${Math.round(n / 1000)}k`;
  const range =
    job.salary_min && job.salary_max
      ? `${fmt(job.salary_min)}-${fmt(job.salary_max)}`
      : fmt(job.salary_min ?? job.salary_max!);
  return `${range}${job.salary_period ? `/${job.salary_period}` : ""}`;
}

function FitBadge({ label, value }: { label: string; value: string | null }) {
  if (!value) return null;
  const cls = value === "High" ? "badge-high" : value === "Med" ? "badge-med" : "badge-low";
  return (
    <span className={`badge ${cls}`}>
      {label}: {value}
    </span>
  );
}

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
        Paste a LinkedIn/Indeed/Glassdoor URL or the raw job text (if the URL fetch fails — common on
        sites that require login — paste the text instead).
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
        {busy ? <Spinner label="Scoring..." /> : "Confirm & score"}
      </button>
      {error && <p className="error-text small">{error}</p>}
    </div>
  );
}

function JobActions({ entry }: { entry: DigestEntry }) {
  const [feedback, setFeedback] = useState<string | null>(null);
  const [tracked, setTracked] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sendFeedback = async (value: "interested" | "not_interested") => {
    try {
      await api.submitFeedback(entry.id, value);
      setFeedback(value);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const track = async () => {
    try {
      await api.createApplication(entry.id);
      setTracked(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="card-section form-actions">
      <button className="ghost small" disabled={feedback === "interested"} onClick={() => sendFeedback("interested")}>
        Interested
      </button>
      <button
        className="ghost small"
        disabled={feedback === "not_interested"}
        onClick={() => sendFeedback("not_interested")}
      >
        Not interested
      </button>
      <button className="ghost small" disabled={tracked} onClick={track}>
        {tracked ? "Tracking" : "Track application"}
      </button>
      {error && <span className="error-text small">{error}</span>}
    </div>
  );
}

export default function DigestPage() {
  const [entries, setEntries] = useState<DigestEntry[]>([]);
  const [pending, setPending] = useState<Job[]>([]);

  const refresh = () => api.getDigestToday().then((r) => setEntries(r.entries));

  useEffect(() => {
    refresh();
  }, []);

  return (
    <div>
      <h1>Today's digest</h1>
      <p className="muted" style={{ marginBottom: "18px" }}>
        Ranked jobs worth your attention today.
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

      {entries.length === 0 && pending.length === 0 && (
        <p className="empty-state">No jobs yet — add one above.</p>
      )}

      {entries.map((entry) => (
        <div className="card" key={entry.id}>
          <div className="card-header">
            <div>
              <div className="card-title">
                {entry.is_favorite_company && <span className="star">★ </span>}
                <Link to={`/jobs/${entry.id}`}>
                  {entry.title} — {entry.company}
                </Link>
              </div>
              <div className="card-meta">
                {entry.location ?? "location unknown"}
                {salaryText(entry) ? ` · ${salaryText(entry)}` : ""} · {daysAgo(entry.first_seen_at)}
              </div>
            </div>
          </div>

          {entry.summary_text && <p>{entry.summary_text}</p>}

          <div className="badge-row">
            <FitBadge label="Interest" value={entry.interest_fit} />
            <FitBadge label="Candidate fit" value={entry.candidate_fit} />
          </div>

          {entry.rationale_text && (
            <div className="muted small">
              <Markdown text={entry.rationale_text} />
            </div>
          )}

          <JobActions entry={entry} />
        </div>
      ))}
    </div>
  );
}
