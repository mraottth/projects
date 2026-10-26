import { ScoredJob } from "../api";

export type RecencyFilter = "all" | "1" | "3" | "7" | "30";

const FIT_OPTIONS = ["High", "Med", "Low"];

export interface JobFilterState {
  interestFilter: string[];
  candidateFilter: string[];
  favoritesOnly: boolean;
  source: string[];
  minSalary: string;
  recency: RecencyFilter;
  location: string[];
  search: string;
}

export const DEFAULT_JOB_FILTERS: JobFilterState = {
  interestFilter: [],
  candidateFilter: [],
  favoritesOnly: false,
  source: [],
  minSalary: "",
  recency: "all",
  location: [],
  search: "",
};

export function filterJobs(jobs: ScoredJob[], f: JobFilterState): ScoredJob[] {
  const minSalary = f.minSalary.trim() ? Number(f.minSalary) : null;
  const recencyDays = f.recency === "all" ? null : Number(f.recency);
  const searchQuery = f.search.trim().toLowerCase();

  return jobs.filter((job) => {
    if (f.interestFilter.length > 0 && !f.interestFilter.includes(job.interest_fit ?? "")) return false;
    if (f.candidateFilter.length > 0 && !f.candidateFilter.includes(job.candidate_fit ?? "")) return false;
    if (f.favoritesOnly && !job.is_favorite) return false;
    if (f.source.length > 0 && !f.source.includes(job.source_name ?? "Unknown source")) return false;
    if (f.location.length > 0 && !f.location.includes(job.location ?? "Unknown location")) return false;

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
  if (f.interestFilter.length) n++;
  if (f.candidateFilter.length) n++;
  if (f.favoritesOnly) n++;
  if (f.source.length) n++;
  if (f.minSalary.trim()) n++;
  if (f.recency !== "all") n++;
  if (f.location.length) n++;
  if (f.search.trim()) n++;
  return n;
}

function MultiSelectGroup({
  label,
  options,
  selected,
  onChange,
}: {
  label: string;
  options: string[];
  selected: string[];
  onChange: (values: string[]) => void;
}) {
  if (options.length === 0) return null;

  const toggle = (value: string) => {
    onChange(selected.includes(value) ? selected.filter((v) => v !== value) : [...selected, value]);
  };

  return (
    <div className="fsection">
      <span className="flabel">{label}</span>
      <div className="multiselect-options">
        {options.map((opt) => (
          <label key={opt} className="multiselect-option">
            <input type="checkbox" checked={selected.includes(opt)} onChange={() => toggle(opt)} />
            <span>{opt}</span>
          </label>
        ))}
      </div>
    </div>
  );
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
  const locations = Array.from(
    new Set(jobs.map((j) => j.location).filter((l): l is string => !!l)),
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

      <MultiSelectGroup
        label="Interest fit"
        options={FIT_OPTIONS}
        selected={filters.interestFilter}
        onChange={(v) => set("interestFilter", v)}
      />

      <MultiSelectGroup
        label="Fit"
        options={FIT_OPTIONS}
        selected={filters.candidateFilter}
        onChange={(v) => set("candidateFilter", v)}
      />

      <MultiSelectGroup
        label="Source"
        options={sources}
        selected={filters.source}
        onChange={(v) => set("source", v)}
      />

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

      <MultiSelectGroup
        label="Location"
        options={locations}
        selected={filters.location}
        onChange={(v) => set("location", v)}
      />

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

      <div className="fsection" style={{ marginBottom: 0 }}>
        <label className="field inline" style={{ marginBottom: 0 }}>
          <input
            type="checkbox"
            checked={filters.favoritesOnly}
            onChange={(e) => set("favoritesOnly", e.target.checked)}
          />
          <span style={{ margin: 0, fontWeight: 400, color: "var(--ink)" }}>Favorites only</span>
        </label>
      </div>
    </div>
  );
}
