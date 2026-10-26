import { useState } from "react";
import { Link } from "react-router-dom";
import { api, ApiError, ScoredJob } from "../api";
import { Markdown } from "./Markdown";

export function daysAgo(iso: string): string {
  const days = Math.floor((Date.now() - new Date(iso).getTime()) / 86_400_000);
  if (days <= 0) return "today";
  if (days === 1) return "1 day ago";
  return `${days} days ago`;
}

export function salaryText(job: ScoredJob): string | null {
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

export function JobActions({ job }: { job: ScoredJob }) {
  const [feedback, setFeedback] = useState<string | null>(null);
  const [tracked, setTracked] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sendFeedback = async (value: "interested" | "not_interested" | "maybe") => {
    setBusy(true);
    setError(null);
    try {
      await api.submitFeedback(job.id, value);
      setFeedback(value);
      if (value === "interested") {
        try {
          await api.createApplication(job.id);
        } catch (e) {
          if (!(e instanceof ApiError && e.status === 409)) throw e;
        }
        setTracked(true);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card-section form-actions">
      <button
        className="ghost small"
        disabled={busy || feedback === "interested"}
        onClick={() => sendFeedback("interested")}
      >
        Interested
      </button>
      <button className="ghost small" disabled={busy || feedback === "maybe"} onClick={() => sendFeedback("maybe")}>
        Maybe
      </button>
      <button
        className="ghost small"
        disabled={busy || feedback === "not_interested"}
        onClick={() => sendFeedback("not_interested")}
      >
        Not interested
      </button>
      {tracked && <span className="muted small">Added to My Jobs — generate docs from the job page.</span>}
      {error && <span className="error-text small">{error}</span>}
    </div>
  );
}

export function ScoredJobCard({ job }: { job: ScoredJob }) {
  return (
    <div className="card">
      <div className="card-header">
        <div>
          <div className="card-title">
            {!!job.is_favorite && <span className="star">★ </span>}
            <Link to={`/jobs/${job.id}`}>
              {job.title} — {job.company}
            </Link>
          </div>
          <div className="card-meta">
            {job.location ?? "location unknown"}
            {salaryText(job) ? ` · ${salaryText(job)}` : ""} · {daysAgo(job.first_seen_at)}
          </div>
        </div>
      </div>

      {job.summary_text && <p>{job.summary_text}</p>}

      <div className="badge-row">
        <FitBadge label="Interest" value={job.interest_fit} />
        <FitBadge label="Candidate fit" value={job.candidate_fit} />
        {!job.interest_fit && !job.candidate_fit && <span className="muted small">Not yet scored</span>}
      </div>

      {job.rationale_text && (
        <div className="muted small">
          <Markdown text={job.rationale_text} />
        </div>
      )}

      <JobActions job={job} />
    </div>
  );
}
