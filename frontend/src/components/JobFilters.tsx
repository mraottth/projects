import { ScoredJob } from "../api";

export type FitFilter = "all" | "High" | "Med" | "Low";
export type RecencyFilter = "all" | "1" | "3" | "7" | "30";

export interface JobFilterState {
  interestFilter: FitFilter;
  candidateFilter: FitFilter;
  favoritesOnly: boolean;
  source: string;
  minSalary: string;
  recency: RecencyFilter;
  location: string;
  search: string;
}

export const DEFAULT_JOB_FILTERS: JobFilterState = {
  interestFilter: "all",
  candidateFilter: "all",
  favoritesOnly: false,
  source: "all",
  minSalary: "",
  recency: "all",
  location: "",
  search: "",
};

export function filterJobs(jobs: ScoredJob[], f: JobFilterState): ScoredJob[] {
  const minSalary = f.minSalary.trim() ? Number(f.minSalary) : null;
  const recencyDays = f.recency === "all" ? null : Number(f.recency);
  const locationQuery = f.location.trim().toLowerCase();
  const searchQuery = f.search.trim().toLowerCase();

  return jobs.filter((job) => {
    if (f.interestFilter !== "all" && job.interest_fit !== f.interestFilter) return false;
    if (f.candidateFilter !== "all" && job.candidate_fit !== f.candidateFilter) return false;
    if (f.favoritesOnly && !job.is_favorite) return false;
    if (f.source !== "all" && (job.source_name ?? "Unknown source") !== f.source) return false;

    if (minSalary !== null) {
      const jobSalary = job.salary_max ?? job.salary_min;
      if (jobSalary == null || jobSalary < minSalary) return false;
    }

    if (recencyDays !== null) {
      const dateStr = job.posted_at ?? job.first_seen_at;
      if (!dateStr) return false;
      const days = (Date.now() - new Date(dateStr).getTime()) / 86_400_000;
      if (days > recencyDays) return false;
    }

    if (locationQuery && !(job.location ?? "").toLowerCase().includes(locationQuery)) return false;

    if (searchQuery) {
      const haystack = [job.title, job.company, job.location, job.summary_text, job.description_text]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      if (!haystack.includes(searchQuery)) return false;
    }

    return true;
  });
}

function countActive(f: JobFilterState): number {
  let n = 0;
  if (f.interestFilter !== "all") n++;
  if (f.candidateFilter !== "all") n++;
  if (f.favoritesOnly) n++;
  if (f.source !== "all") n++;
  if (f.minSalary.trim()) n++;
  if (f.recency !== "all") n++;
  if (f.location.trim()) n++;
  if (f.search.trim()) n++;
  return n;
}

export function JobFilterBar({
  jobs,
  filters,
  onChange,
}: {
  jobs: ScoredJob[];
  filters: JobFilterState;
  onChange: (f: JobFilterState) => void;
}) {
  const sources = Array.from(
    new Set(jobs.map((j) => j.source_name).filter((s): s is string => !!s)),
  ).sort();

  const set = <K extends keyof JobFilterState>(key: K, value: JobFilterState[K]) =>
    onChange({ ...filters, [key]: value });

  const active = countActive(filters);

  return (
    <div className="card filters">
      <div className="filters-head">
        <h2 style={{ fontSize: "1rem", margin: 0 }}>Filters{active ? ` (${active})` : ""}</h2>
        {active > 0 && (
          <button className="link small" onClick={() => onChange(DEFAULT_JOB_FILTERS)}>
            Clear all
          </button>
        )}
      </div>

      <div className="fsection">
        <span className="flabel">Search</span>
        <input
          placeholder="Title, company, location, full description..."
          value={filters.search}
          onChange={(e) => set("search", e.target.value)}
          style={{ width: "100%" }}
        />
      </div>

      <div className="fsection">
        <span className="flabel">Interest fit</span>
        <select
          value={filters.interestFilter}
          onChange={(e) => set("interestFilter", e.target.value as FitFilter)}
          style={{ width: "100%" }}
        >
          <option value="all">All</option>
          <option value="High">High</option>
          <option value="Med">Med</option>
          <option value="Low">Low</option>
        </select>
      </div>

      <div className="fsection">
        <span className="flabel">Candidate fit</span>
        <select
          value={filters.candidateFilter}
          onChange={(e) => set("candidateFilter", e.target.value as FitFilter)}
          style={{ width: "100%" }}
        >
          <option value="all">All</option>
          <option value="High">High</option>
          <option value="Med">Med</option>
          <option value="Low">Low</option>
        </select>
      </div>

      <div className="fsection">
        <span className="flabel">Source</span>
        <select value={filters.source} onChange={(e) => set("source", e.target.value)} style={{ width: "100%" }}>
          <option value="all">All</option>
          {sources.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      </div>

      <div className="fsection">
        <span className="flabel">Posted</span>
        <select
          value={filters.recency}
          onChange={(e) => set("recency", e.target.value as RecencyFilter)}
          style={{ width: "100%" }}
        >
          <option value="all">Any time</option>
          <option value="1">Last 24 hours</option>
          <option value="3">Last 3 days</option>
          <option value="7">Last 7 days</option>
          <option value="30">Last 30 days</option>
        </select>
      </div>

      <div className="fsection">
        <span className="flabel">Location</span>
        <input
          placeholder="e.g. Remote, NYC"
          value={filters.location}
          onChange={(e) => set("location", e.target.value)}
          style={{ width: "100%" }}
        />
      </div>

      <div className="fsection">
        <span className="flabel">Minimum salary</span>
        <input
          type="number"
          placeholder="e.g. 120000"
          value={filters.minSalary}
          onChange={(e) => set("minSalary", e.target.value)}
          style={{ width: "100%" }}
        />
      </div>

      <label className="field inline" style={{ marginBottom: 0 }}>
        <input
          type="checkbox"
          checked={filters.favoritesOnly}
          onChange={(e) => set("favoritesOnly", e.target.checked)}
        />
        <span style={{ margin: 0, fontWeight: 400, color: "var(--ink)" }}>Favorites only</span>
      </label>
    </div>
  );
}
