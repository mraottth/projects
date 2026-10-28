import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, RankedJob } from "../api";
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

export default function AllJobsPage() {
  const [jobs, setJobs] = useState<RankedJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState<JobFilterState>(DEFAULT_JOB_FILTERS);

  useEffect(() => {
    api
      .getRankedJobs()
      .then(setJobs)
      .finally(() => setLoading(false));
  }, []);

  const filtered = filterJobs(jobs, filters);

  return (
    <div>
      <h1>All Jobs</h1>
      <p className="muted" style={{ marginBottom: "18px" }}>
        Every active posting from the companies and job boards you follow, ranked by interest and
        candidate fit first — not just what made today's New &amp; Noteworthy list.
      </p>

      {loading && <p className="muted">Loading...</p>}
      {!loading && jobs.length === 0 && (
        <div className="card empty-card">
          <p>No jobs yet.</p>
          <p className="muted small">
            Jobs show up here once your sources are checked — that happens every morning, or right
            away with "Run now" on Manage Sources.
          </p>
          <Link className="primary button-link" to="/sources">
            Manage sources
          </Link>
        </div>
      )}

      {jobs.length > 0 && (
        <div className="list-layout">
          <div className="list-main">
            <JobListHeader
              count={filtered.length}
              total={jobs.length}
              sort={filters.sort}
              onSort={(sort) => setFilters({ ...filters, sort })}
            />
            {filtered.length === 0 && <FilteredEmptyState onClear={() => setFilters(NO_JOB_FILTERS)} />}
            {filtered.map((job) => (
              <ScoredJobCard key={job.id} job={job} />
            ))}
          </div>
          <aside className="list-side">
            <JobFilterBar jobs={jobs} filters={filters} onChange={setFilters} />
          </aside>
        </div>
      )}
    </div>
  );
}
