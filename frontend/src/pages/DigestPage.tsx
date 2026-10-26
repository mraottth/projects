import { useEffect, useState } from "react";
import { api, DigestEntry } from "../api";
import { DEFAULT_JOB_FILTERS, JobFilterBar, JobFilterState, filterJobs } from "../components/JobFilters";
import { ScoredJobCard } from "../components/ScoredJobCard";

export default function DigestPage() {
  const [entries, setEntries] = useState<DigestEntry[]>([]);
  const [filters, setFilters] = useState<JobFilterState>(DEFAULT_JOB_FILTERS);

  useEffect(() => {
    api.getDigestToday().then((r) => setEntries(r.entries));
  }, []);

  const filtered = filterJobs(entries, filters);

  return (
    <div>
      <h1>Today's digest</h1>
      <p className="muted" style={{ marginBottom: "18px" }}>
        New and noteworthy — the best of what showed up today. For everything your tracked
        sources have turned up, see My Sources.
      </p>

      {entries.length === 0 && <p className="empty-state">Nothing in today's digest yet.</p>}

      {entries.length > 0 && <JobFilterBar jobs={entries} filters={filters} onChange={setFilters} />}

      {entries.length > 0 && filtered.length === 0 && (
        <p className="empty-state">No jobs match these filters.</p>
      )}

      {filtered.map((entry) => (
        <ScoredJobCard key={entry.id} job={entry} />
      ))}
    </div>
  );
}
