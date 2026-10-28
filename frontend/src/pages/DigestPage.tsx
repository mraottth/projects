import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, DigestEntry } from "../api";
import {
  DEFAULT_JOB_FILTERS,
  FilteredEmptyState,
  JobFilterBar,
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

  useEffect(() => {
    api
      .getDigestToday()
      .then((r) => setEntries(r.entries))
      .finally(() => setLoading(false));
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
      <p className="muted" style={{ marginBottom: "18px" }}>
        The best of what showed up today, ranked by interest and candidate fit first. For
        everything your tracked sources have turned up, see All Jobs.
      </p>

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
            New postings are checked and scored every morning. In the meantime, browse everything
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
