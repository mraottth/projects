import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, ApiError, DigestEntry, DigestStatus, RunProgress, runProgressText } from "../api";
import { Spinner } from "../components/Spinner";
import {
  DEFAULT_JOB_FILTERS,
  FilteredEmptyState,
  JobFilterBar,
  JobListHeader,
  JobFilterState,
  NO_JOB_FILTERS,
  filterJobs,
} from "../components/JobFilters";
import { ScoredJobCard } from "../components/ScoredJobCard";

export default function DigestPage() {
  const [entries, setEntries] = useState<DigestEntry[]>([]);
  const [filters, setFilters] = useState<JobFilterState>(DEFAULT_JOB_FILTERS);
  const [hasResume, setHasResume] = useState(true);
  const [loading, setLoading] = useState(true);

  const loadEntries = () =>
    api
      .getDigestToday()
      .then((r) => setEntries(r.entries))
      .finally(() => setLoading(false));

  useEffect(() => {
    loadEntries();
    // Covers anyone who skipped onboarding: nothing gets scored well without a resume.
    api
      .getOnboardingStatus()
      .then((s) => setHasResume(s.has_resume))
      .catch(() => undefined);
  }, []);

  const filtered = filterJobs(entries, filters);

  return (
    <div>
      <h1>New &amp; Noteworthy</h1>
      <p className="muted" style={{ marginBottom: "8px" }}>
        The best of what showed up today, ranked by interest and candidate fit first. For
        everything your tracked sources have turned up, see All Jobs.
      </p>
      {hasResume && <Freshness onRefreshed={loadEntries} />}

      {!hasResume && (
        <div className="card">
          <h2>Finish setting up</h2>
          <p className="muted small">
            Upload your resume and tell the assistant what you're looking for, so it can find and
            score jobs for you.
          </p>
          <Link className="primary onboarding-cta" to="/onboarding">
            Continue setup
          </Link>
        </div>
      )}

      {loading && <p className="muted">Loading...</p>}
      {!loading && entries.length === 0 && hasResume && (
        <div className="card empty-card">
          <p>Nothing new and noteworthy today.</p>
          <p className="muted small">
            New postings are checked and scored every day at 2:00pm ET. In the meantime, browse everything
            your sources have already found, or add more sources to widen the net.
          </p>
          <div className="form-actions">
            <Link className="primary button-link" to="/all-jobs">
              Browse All Jobs
            </Link>
            <Link className="ghost button-link" to="/sources">
              Add more sources
            </Link>
          </div>
        </div>
      )}

      {entries.length > 0 && (
        <div className="list-layout">
          <div className="list-main">
            <JobListHeader
              count={filtered.length}
              total={entries.length}
              sort={filters.sort}
              onSort={(sort) => setFilters({ ...filters, sort })}
            />
            {filtered.length === 0 && <FilteredEmptyState onClear={() => setFilters(NO_JOB_FILTERS)} />}
            {filtered.map((entry) => (
              <ScoredJobCard
                key={entry.id}
                job={entry}
                onDismissed={() => setEntries((prev) => prev.filter((e) => e.id !== entry.id))}
              />
            ))}
          </div>
          <aside className="list-side">
            <JobFilterBar jobs={entries} filters={filters} onChange={setFilters} />
          </aside>
        </div>
      )}
    </div>
  );
}

const STALE_AFTER_MS = 24 * 3600 * 1000;

function ago(iso: string): string {
  const ms = Date.now() - new Date(iso).getTime();
  const minutes = Math.round(ms / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes} minute${minutes === 1 ? "" : "s"} ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} hour${hours === 1 ? "" : "s"} ago`;
  const days = Math.round(hours / 24);
  return `${days} day${days === 1 ? "" : "s"} ago`;
}

function minutesUntil(iso: string): number {
  return Math.max(1, Math.ceil((new Date(iso).getTime() - Date.now()) / 60000));
}

/** "Last updated 2 hours ago · Refresh" (step 5 of D-021): this person's own check, at most every
 * 30 minutes, scoring within today's cap. After a day or more without an update (someone back
 * from a break), it becomes a prominent card instead. */
function Freshness({ onRefreshed }: { onRefreshed: () => void }) {
  const [status, setStatus] = useState<DigestStatus | null>(null);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<RunProgress | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadStatus = () =>
    api
      .getDigestStatus()
      .then(setStatus)
      .catch(() => undefined);

  useEffect(() => {
    loadStatus();
  }, []);

  async function refresh() {
    setRunning(true);
    setProgress(null);
    setError(null);
    setNote(null);
    try {
      const summary = await api.refreshDigest(setProgress);
      const scored = Number(summary.llm_scored_count ?? 0);
      setNote(scored ? `Scored ${scored} new job${scored === 1 ? "" : "s"}.` : "Nothing new to score.");
      onRefreshed();
    } catch (e) {
      setError(e instanceof ApiError || e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
      loadStatus();
    }
  }

  if (!status) return null;
  const capUsed = status.scored_today >= status.daily_score_cap;
  const stale = !status.last_updated || Date.now() - new Date(status.last_updated).getTime() > STALE_AFTER_MS;
  const waitTitle = status.refresh_available_at
    ? `You can refresh again in ${minutesUntil(status.refresh_available_at)} minutes`
    : undefined;
  const button = (primary: boolean) => (
    <button
      className={primary ? "primary" : "link"}
      onClick={refresh}
      disabled={running || !status.can_refresh}
      title={waitTitle}
    >
      {primary ? "Refresh now" : "Refresh"}
    </button>
  );
  const limits = status.over_budget
    ? "This month's budget is used up, so new jobs won't be scored until next month."
    : capUsed
      ? `Scoring resumes tomorrow (${status.scored_today} of ${status.daily_score_cap} used today).`
      : null;

  return (
    <div className={stale && !running ? "card freshness-card" : "freshness"}>
      {running ? (
        <Spinner label={runProgressText(progress, "Finishing up...")} />
      ) : stale ? (
        <>
          <p style={{ margin: 0 }}>
            {status.last_updated
              ? `Your matches were last updated ${ago(status.last_updated)}.`
              : "Your matches haven't been checked yet."}
          </p>
          <div className="form-actions">{button(true)}</div>
        </>
      ) : (
        <span className="muted small">
          Last updated {ago(status.last_updated!)} · {button(false)}
        </span>
      )}
      {!running && waitTitle && !stale && <span className="muted small"> ({waitTitle.toLowerCase()})</span>}
      {!running && limits && <p className="muted small freshness-note">{limits}</p>}
      {!running && note && <p className="muted small freshness-note">{note}</p>}
      {error && <p className="error small">{error}</p>}
    </div>
  );
}

