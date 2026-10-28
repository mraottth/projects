import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, DigestEntry } from "../api";
import { DEFAULT_JOB_FILTERS, JobFilterBar, JobFilterState, filterJobs } from "../components/JobFilters";
import { ScoredJobCard } from "../components/ScoredJobCard";

export default function DigestPage() {
  const [entries, setEntries] = useState<DigestEntry[]>([]);
  const [filters, setFilters] = useState<JobFilterState>(DEFAULT_JOB_FILTERS);
  const [hasResume, setHasResume] = useState(true);

  useEffect(() => {
    api.getDigestToday().then((r) => setEntries(r.entries));
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

      {entries.length === 0 && <p className="empty-state">Nothing new and noteworthy yet.</p>}

      {entries.length > 0 && (
        <div className="list-layout">
          <div className="list-main">
            {filtered.length === 0 && <p className="empty-state">No jobs match these filters.</p>}
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
