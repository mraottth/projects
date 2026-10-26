import { useEffect, useState } from "react";
import { api, DigestEntry, Job } from "../api";

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
  const color = value === "High" ? "green" : value === "Med" ? "orange" : "crimson";
  return (
    <span style={{ color, fontWeight: 600, marginRight: "1rem" }}>
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
      <p className="muted">
        Paste a LinkedIn/Indeed/Glassdoor URL or the raw job text (if the URL fetch fails — common on
        sites that require login — paste the text instead).
      </p>
      <div style={{ display: "flex", gap: "0.5rem", marginBottom: "0.5rem" }}>
        <label>
          <input type="radio" checked={mode === "text"} onChange={() => setMode("text")} /> Raw text
        </label>
        <label>
          <input type="radio" checked={mode === "url"} onChange={() => setMode("url")} /> URL
        </label>
      </div>
      {mode === "text" ? (
        <textarea
          rows={6}
          style={{ width: "100%" }}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Paste the job description here..."
        />
      ) : (
        <input
          style={{ width: "100%" }}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="https://..."
        />
      )}
      <div style={{ marginTop: "0.5rem" }}>
        <button onClick={submit} disabled={busy}>
          {busy ? "Parsing..." : "Parse"}
        </button>
      </div>
      {error && <p style={{ color: "crimson" }}>{error}</p>}
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
      <input
        style={{ width: "100%", marginBottom: "0.4rem" }}
        value={draft.title}
        onChange={(e) => setDraft({ ...draft, title: e.target.value })}
      />
      <input
        style={{ width: "100%", marginBottom: "0.4rem" }}
        value={draft.company}
        onChange={(e) => setDraft({ ...draft, company: e.target.value })}
      />
      <input
        style={{ width: "100%", marginBottom: "0.4rem" }}
        value={draft.location ?? ""}
        onChange={(e) => setDraft({ ...draft, location: e.target.value })}
        placeholder="Location"
      />
      <button onClick={confirm} disabled={busy}>
        {busy ? "Scoring..." : "Confirm & score"}
      </button>
      {error && <p style={{ color: "crimson" }}>{error}</p>}
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
    <div style={{ marginTop: "0.5rem", display: "flex", gap: "0.5rem", alignItems: "center" }}>
      <button disabled={feedback === "interested"} onClick={() => sendFeedback("interested")}>
        Interested
      </button>
      <button disabled={feedback === "not_interested"} onClick={() => sendFeedback("not_interested")}>
        Not interested
      </button>
      <button disabled={tracked} onClick={track}>
        {tracked ? "Tracking" : "Track application"}
      </button>
      {error && <span style={{ color: "crimson" }}>{error}</span>}
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
        <p className="muted">No jobs yet — add one above.</p>
      )}

      {entries.map((entry) => (
        <div className="card" key={entry.id}>
          <h3>
            {entry.title} — {entry.company}
          </h3>
          <p className="muted">
            {entry.location ?? "location unknown"}
            {salaryText(entry) ? ` · ${salaryText(entry)}` : ""} · {daysAgo(entry.first_seen_at)}
          </p>
          <p>{entry.summary_text}</p>
          <div>
            <FitBadge label="Interest" value={entry.interest_fit} />
            <FitBadge label="Candidate fit" value={entry.candidate_fit} />
          </div>
          {entry.rationale_text && <p className="muted">{entry.rationale_text}</p>}
          <JobActions entry={entry} />
        </div>
      ))}
    </div>
  );
}
