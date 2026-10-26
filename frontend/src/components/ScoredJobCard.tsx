import { useLayoutEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, ApiError, ScoredJob } from "../api";
import { Markdown } from "./Markdown";

function daysAgo(iso: string): string {
  const days = Math.floor((Date.now() - new Date(iso).getTime()) / 86_400_000);
  if (days <= 0) return "today";
  if (days === 1) return "1 day ago";
  return `${days} days ago`;
}

export function salaryText(job: ScoredJob): string {
  if (!job.salary_min && !job.salary_max) return "N/A";
  const fmt = (n: number) => `$${Math.round(n / 1000)}k`;
  const range =
    job.salary_min && job.salary_max
      ? `${fmt(job.salary_min)}-${fmt(job.salary_max)}`
      : fmt(job.salary_min ?? job.salary_max!);
  return `${range}${job.salary_period ? `/${job.salary_period}` : ""}`;
}

// The job description's own posted date, not when our pipeline first saw it -- falls back to
// "N/A" rather than first_seen_at, since "when did our scraper happen to find this" isn't what a
// "date posted" label should mean.
export function postedText(job: ScoredJob): string {
  return job.posted_at ? daysAgo(job.posted_at) : "N/A";
}

function RationaleBlock({ text }: { text: string }) {
  const [expanded, setExpanded] = useState(false);
  const [overflowing, setOverflowing] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    if (ref.current) setOverflowing(ref.current.scrollHeight > ref.current.clientHeight + 1);
  }, [text]);

  return (
    <div className="muted small">
      <div ref={ref} className={expanded ? undefined : "clamp-2"}>
        <Markdown text={text} />
      </div>
      {overflowing && (
        <button className="link small" onClick={() => setExpanded((e) => !e)}>
          {expanded ? "Show less" : "Show more"}
        </button>
      )}
    </div>
  );
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
            Location: {job.location ?? "N/A"} · Salary: {salaryText(job)} · Posted: {postedText(job)}
            {job.source_name && ` · ${job.source_name}`}
          </div>
        </div>
      </div>

      {job.summary_text && <p>{job.summary_text}</p>}

      <div className="badge-row">
        <FitBadge label="Interest" value={job.interest_fit} />
        <FitBadge label="Candidate fit" value={job.candidate_fit} />
        {!job.interest_fit && !job.candidate_fit && <span className="muted small">Not yet scored</span>}
      </div>

      {job.rationale_text && <RationaleBlock text={job.rationale_text} />}

      <JobActions job={job} />
    </div>
  );
}
