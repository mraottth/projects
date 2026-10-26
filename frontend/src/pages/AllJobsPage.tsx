import { useEffect, useState } from "react";
import { api, RankedJob } from "../api";
import { DEFAULT_JOB_FILTERS, JobFilterBar, JobFilterState, filterJobs } from "../components/JobFilters";
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
        Every active posting from the companies and job boards you follow, best first — not just
        what made today's New &amp; Noteworthy list.
      </p>

      {jobs.length > 0 && <JobFilterBar jobs={jobs} filters={filters} onChange={setFilters} />}

      {loading && <p className="muted">Loading...</p>}
      {!loading && jobs.length === 0 && (
        <p className="empty-state">
          No jobs yet — add a source on the Manage Sources page, or run the pipeline to poll what's
          already tracked.
        </p>
      )}
      {!loading && jobs.length > 0 && filtered.length === 0 && (
        <p className="empty-state">No jobs match these filters.</p>
      )}
      {filtered.map((job) => (
        <ScoredJobCard key={job.id} job={job} />
      ))}
    </div>
  );
}
