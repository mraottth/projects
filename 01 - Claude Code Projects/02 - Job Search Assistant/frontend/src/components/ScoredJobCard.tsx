import { useLayoutEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { JobLink } from "./JobLink";
import { api, ApiError, APPLICATION_STATUS_LABELS, ApplicationStatus, ScoredJob } from "../api";
import { Markdown } from "./Markdown";
import { Spinner } from "./Spinner";
import { DEMO } from "../demo";

function daysAgo(iso: string): string {
  const days = Math.floor((Date.now() - new Date(iso).getTime()) / 86_400_000);
  if (days <= 0) return "today";
  if (days === 1) return "1 day ago";
  return `${days} days ago`;
}

interface SalaryFields {
  salary_min: number | null;
  salary_max: number | null;
  salary_period: string | null;
  salary_currency?: string | null;
}

const CURRENCY_SYMBOLS: Record<string, string> = { USD: "$", CAD: "CA$", GBP: "£", EUR: "€" };
// Yearly is the assumed default, so it gets no suffix; anything else says its period.
const PERIOD_LABELS: Record<string, string> = { year: "", hour: "/hr" };

// "$120k-150k", "$45-60/hr". Hourly figures stay whole dollars: rounding them to thousands
// showed $40/hour as "$0k".
export function salaryText(job: SalaryFields): string {
  if (!job.salary_min && !job.salary_max) return "N/A";
  const symbol = CURRENCY_SYMBOLS[job.salary_currency ?? "USD"] ?? `${job.salary_currency} `;
  const fmt = (n: number) => (n >= 1000 ? `${Math.round(n / 1000)}k` : `${Math.round(n * 100) / 100}`);
  const range =
    job.salary_min && job.salary_max && job.salary_min !== job.salary_max
      ? `${fmt(job.salary_min)}-${fmt(job.salary_max)}`
      : fmt(job.salary_min ?? job.salary_max!);
  const period = job.salary_period ? (PERIOD_LABELS[job.salary_period] ?? `/${job.salary_period}`) : "";
  return `${symbol}${range}${period}`;
}

export function salaryValue(job: Pick<SalaryFields, "salary_min" | "salary_max">): number {
  return job.salary_max ?? job.salary_min ?? -1;
}

// The job description's own posted date, not when our pipeline first saw it -- falls back to
// "N/A" rather than first_seen_at, since "when did our scraper happen to find this" isn't what a
// "date posted" label should mean.
export function postedText(job: ScoredJob): string {
  return job.posted_at ? daysAgo(job.posted_at) : "N/A";
}

// Classes that color a rationale's ==i:...== / ==f:...== highlights like the Interest / Fit badges.
export function rationaleClass(interest: string | null, fit: string | null): string {
  const level = (v: string | null) => (v ? v.toLowerCase() : "none");
  return `rationale interest-${level(interest)} fit-${level(fit)}`;
}

function RationaleBlock({ text, className }: { text: string; className: string }) {
  const [expanded, setExpanded] = useState(false);
  const [overflowing, setOverflowing] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    if (ref.current) setOverflowing(ref.current.scrollHeight > ref.current.clientHeight + 1);
  }, [text]);

  return (
    <div className={`muted small ${className}`}>
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

// Best-effort only: guesses a .com domain from the company name and asks Clearbit's free logo
// API for it. No API key, no backend involvement -- just hides itself on a 404/network error, so
// a wrong guess (most non-trivial company names) silently shows nothing rather than a broken image.
function guessDomain(company: string): string | null {
  const cleaned = company.toLowerCase().replace(/[^a-z0-9]+/g, "");
  return cleaned ? `${cleaned}.com` : null;
}

function CompanyLogo({ company }: { company: string }) {
  const [failed, setFailed] = useState(false);
  const domain = guessDomain(company);
  if (!domain || failed) return null;
  return (
    <img
      src={`https://logo.clearbit.com/${domain}?size=64`}
      alt=""
      className="company-logo"
      onError={() => setFailed(true)}
    />
  );
}

export function FitBadge({ label, value }: { label: string; value: string | null }) {
  if (!value) return null;
  const cls = value === "High" ? "badge-high" : value === "Med" ? "badge-med" : "badge-low";
  return (
    <span className={`badge ${cls}`}>
      {label}: {value}
    </span>
  );
}

export function JobActions({
  job,
  onDismissed,
  onTracked,
}: {
  job: ScoredJob;
  onDismissed?: () => void;
  onTracked?: () => void;
}) {
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
        onTracked?.();
      }
      if (value === "not_interested") onDismissed?.();
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

export type ChipFilterKey = "location" | "organization";

// "Recidiviz" and "recidiviz " are the same name; punctuation and case don't count.
function sameName(a: string, b: string): boolean {
  const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, "");
  return norm(a) === norm(b);
}

/**
 * A meta chip that, given `onFilter`, applies (or clears) that filter on the list it's in. Shown
 * pressed while its filter is on.
 */
function FilterChip({
  className,
  value,
  filterKey,
  onFilter,
  active,
}: {
  className: string;
  value: string;
  filterKey: ChipFilterKey;
  onFilter?: (key: ChipFilterKey, value: string) => void;
  active?: boolean;
}) {
  if (!onFilter) return <span className={`meta-chip ${className}`}>{value}</span>;
  const what = filterKey === "location" ? "location" : "organization";
  return (
    <button
      type="button"
      className={`meta-chip meta-chip-button ${className}${active ? " on" : ""}`}
      aria-pressed={!!active}
      title={active ? `Stop filtering by this ${what}` : `Show only jobs in this ${what}`}
      onClick={() => onFilter(filterKey, value)}
    >
      {value}
    </button>
  );
}

/** "Score this job": one live score, about a cent (POST /api/jobs/{id}/score). */
export function ScoreButton({
  jobId,
  onScored,
  label = "Score this job",
}: {
  jobId: number;
  onScored: (score: Partial<ScoredJob>) => void;
  label?: string;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  return (
    <>
      <button
        className="ghost small score-now"
        disabled={busy}
        title="Has Claude rate this job now (about a cent)"
        onClick={async () => {
          setBusy(true);
          setError(null);
          try {
            onScored(await api.scoreJob(jobId));
          } catch (e) {
            setError(e instanceof Error ? e.message : String(e));
          } finally {
            setBusy(false);
          }
        }}
      >
        {busy ? <Spinner label="Scoring..." /> : label}
      </button>
      {error && <span className="error-text small">{error}</span>}
    </>
  );
}

export function ScoredJobCard({
  job: given,
  onDismissed,
  onFilter,
  activeFilters,
}: {
  job: ScoredJob;
  onDismissed?: () => void;
  onFilter?: (key: ChipFilterKey, value: string) => void;
  activeFilters?: { location: string[]; organization: string[] };
}) {
  // Where it stands on My Jobs (asked 2026-11-10); marking it Interested here adds it as Drafting.
  const [status, setStatus] = useState<ApplicationStatus | null>(given.application_status ?? null);
  // A score asked for from this card ("Score this job", asked 2026-11-12) shows here at once.
  const [score, setScore] = useState<Partial<ScoredJob> | null>(null);
  const job = score ? { ...given, ...score } : given;
  return (
    <div className="card">
      <div className="card-header">
        <div className="card-header-main">
          <CompanyLogo company={job.company} />
          <div>
            <div className="card-title">
              <JobLink jobId={job.id}>
                {job.title} — {job.company}
              </JobLink>
            </div>
            <div className="meta-chips">
              <FilterChip
                className="meta-chip-org"
                value={job.company}
                filterKey="organization"
                onFilter={onFilter}
                active={activeFilters?.organization.includes(job.company)}
              />
              {job.location ? (
                <FilterChip
                  className="meta-chip-location"
                  value={job.location}
                  filterKey="location"
                  onFilter={onFilter}
                  active={activeFilters?.location.includes(job.location)}
                />
              ) : (
                <span className="meta-chip meta-chip-location">Location: N/A</span>
              )}
              <span className="meta-chip meta-chip-salary">Salary: {salaryText(job)}</span>
              <span className="meta-chip meta-chip-posted">Date Posted: {postedText(job)}</span>
              {/* Where it was found, only for a job board (asked 2026-11-06): an organization's own careers page
                  adds nothing to the organization chip. A board's own postings don't repeat its name either. */}
              {!!job.source_is_board && job.source_name && !sameName(job.source_name, job.company) && (
                <span className="meta-chip meta-chip-source">{job.source_name}</span>
              )}
            </div>
          </div>
        </div>
      </div>

      {job.summary_text && <p>{job.summary_text}</p>}

      <div className="badge-row">
        <FitBadge label="Interest" value={job.interest_fit} />
        <FitBadge label="Fit" value={job.candidate_fit} />
        {!job.interest_fit && !job.candidate_fit && (
          <>
            <span className="muted small">
              {"screened_out" in job && job.screened_out
                ? "Not scored: its title looked like a clear mismatch"
                : "Not yet scored"}
            </span>
            {!DEMO && <ScoreButton jobId={job.id} onScored={setScore} />}
          </>
        )}
      </div>

      {job.rationale_text && (
        <RationaleBlock text={job.rationale_text} className={rationaleClass(job.interest_fit, job.candidate_fit)} />
      )}

      {/* On My Jobs already: its status there replaces the Interested / Maybe / Not interested
          buttons (asked 2026-11-10). Marking it Interested here switches to "Drafting". */}
      {status ? (
        <div className="card-section form-actions">
          <Link to="/tracker" className={`pill pill-${status} tracked-pill`} title="Open My Jobs">
            On My Jobs · {APPLICATION_STATUS_LABELS[status]}
          </Link>
        </div>
      ) : (
        <JobActions job={job} onDismissed={onDismissed} onTracked={() => setStatus("saved")} />
      )}
    </div>
  );
}
