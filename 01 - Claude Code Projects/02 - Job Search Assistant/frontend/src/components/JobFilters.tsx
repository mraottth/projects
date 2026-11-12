import { ScoredJob } from "../api";

export type RecencyFilter = "all" | "1" | "3" | "7" | "30";

const FIT_OPTIONS = ["High", "Med", "Low"];

export interface JobFilterState {
  interestFilter: string[];
  candidateFilter: string[];
  source: string[];
  minSalary: string;
  recency: RecencyFilter;
  location: string[];
  search: string;
  // Jobs the daily cap hasn't scored yet have no fit at all, so the fit filters can't judge them;
  // this alone decides whether they show.
  includeUnscored: boolean;
  organization: string[];
  // Jobs your dealbreakers (location, salary) hide on All Jobs. A setting, not a filter, so
  // "Clear all" leaves it off.
  includeDealbreakers: boolean;
  // Jobs the title screen set aside as clear mismatches (core/screening.py): never scored, hidden
  // unless asked for. A setting like the dealbreakers one.
  includeScreenedOut: boolean;
  // Only the set-aside jobs (Settings' Title screen card links to All Jobs?set_aside=1).
  onlyScreenedOut?: boolean;
  sort: SortKey;
}

// "best" keeps the server's order: interest + fit together, then most recent, then salary.
export type SortKey = "best" | "interest" | "fit" | "posted" | "salary";

const SORT_OPTIONS: { value: SortKey; label: string }[] = [
  { value: "best", label: "Best match" },
  { value: "interest", label: "Interest" },
  { value: "fit", label: "Fit" },
  { value: "posted", label: "Date posted" },
  { value: "salary", label: "Salary" },
];

export const DEFAULT_JOB_FILTERS: JobFilterState = {
  interestFilter: ["High", "Med"],
  candidateFilter: ["High", "Med"],
  source: [],
  minSalary: "",
  recency: "all",
  location: [],
  search: "",
  includeUnscored: true,
  organization: [],
  includeDealbreakers: false,
  includeScreenedOut: false,
  sort: "best",
};

// Truly no filters -- unlike DEFAULT_JOB_FILTERS, which already hides Low fits.
export const NO_JOB_FILTERS: JobFilterState = {
  interestFilter: [],
  candidateFilter: [],
  source: [],
  minSalary: "",
  recency: "all",
  location: [],
  search: "",
  includeUnscored: true,
  organization: [],
  includeDealbreakers: false,
  includeScreenedOut: false,
  sort: "best",
};

// Above the job list: how many jobs match, and the sort order.
export function JobListHeader({
  count,
  total,
  sort,
  onSort,
}: {
  count: number;
  total: number;
  sort: SortKey;
  onSort: (sort: SortKey) => void;
}) {
  return (
    <div className="list-head">
      <span className="muted small">
        {count === total ? `${total} ${total === 1 ? "job" : "jobs"}` : `${count} of ${total} jobs`}
      </span>
      <label className="list-sort small">
        Sort by
        <span className="select-wrap">
          <select value={sort} onChange={(e) => onSort(e.target.value as SortKey)}>
            {SORT_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </span>
      </label>
    </div>
  );
}

// Shown in place of the list when jobs exist but the filters hide all of them.
export function FilteredEmptyState({ onClear }: { onClear: () => void }) {
  return (
    <div className="card empty-card">
      <p>No jobs match these filters.</p>
      <p className="muted small">
        The default filters hide Low fits.
      </p>
      <button className="primary" onClick={onClear}>
        Clear all filters
      </button>
    </div>
  );
}

// Highest first (scores are 0-100; the other score breaks ties); a job missing the value (unscored, no salary, unknown date) goes last. Ties keep
// the incoming order (Array.sort is stable), which is the "best match" ranking.
function sortValue(job: ScoredJob, sort: SortKey): number | null {
  switch (sort) {
    case "interest":
      return job.interest_score == null ? null : job.interest_score * 1000 + (job.candidate_fit_score ?? 0);
    case "fit":
      return job.candidate_fit_score == null ? null : job.candidate_fit_score * 1000 + (job.interest_score ?? 0);
    case "posted":
      return job.posted_at ? new Date(job.posted_at).getTime() || null : null;
    case "salary": {
      const pay = job.salary_max ?? job.salary_min;
      // Hourly pay compared as a full-time year (2,080 hours), so $60/hr ranks near $125k.
      return pay == null ? null : job.salary_period === "hour" ? pay * 2080 : pay;
    }
    default:
      return 0;
  }
}

export function sortJobs(jobs: ScoredJob[], sort: SortKey): ScoredJob[] {
  if (sort === "best") return jobs;
  return [...jobs].sort((a, b) => {
    const va = sortValue(a, sort);
    const vb = sortValue(b, sort);
    if (va === null || vb === null) return va === vb ? 0 : va === null ? 1 : -1;
    return vb - va;
  });
}

export function filterJobs(jobs: ScoredJob[], f: JobFilterState): ScoredJob[] {
  const minSalary = f.minSalary.trim() ? Number(f.minSalary) : null;
  const recencyDays = f.recency === "all" ? null : Number(f.recency);
  const searchQuery = f.search.trim().toLowerCase();

  const filtered = jobs.filter((job) => {
    if (job.dealbreakers?.length && !f.includeDealbreakers) return false;
    if (job.screened_out && !f.includeScreenedOut) return false;
    if (f.onlyScreenedOut && !job.screened_out) return false;
    if (!job.interest_fit && !job.candidate_fit) {
      if (!f.includeUnscored) return false;
    } else {
      if (f.interestFilter.length > 0 && !f.interestFilter.includes(job.interest_fit ?? "")) return false;
      if (f.candidateFilter.length > 0 && !f.candidateFilter.includes(job.candidate_fit ?? "")) return false;
    }
    if (f.source.length > 0 && !f.source.includes(job.source_name ?? "Unknown source")) return false;
    if (f.location.length > 0 && !f.location.includes(job.location ?? "Unknown location")) return false;
    if (f.organization.length > 0 && !f.organization.includes(job.company)) return false;

    if (minSalary !== null) {
      const pay = job.salary_max ?? job.salary_min;
      const jobSalary = pay != null && job.salary_period === "hour" ? pay * 2080 : pay;
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
  return sortJobs(filtered, f.sort);
}

function countActive(f: JobFilterState): number {
  let n = 0;
  if (f.interestFilter.length) n++;
  if (f.candidateFilter.length) n++;
  if (f.source.length) n++;
  if (f.minSalary.trim()) n++;
  if (f.recency !== "all") n++;
  if (f.location.length) n++;
  if (f.organization.length) n++;
  if (f.search.trim()) n++;
  if (!f.includeUnscored) n++;
  return n;
}

function ChipMultiSelect({
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
  const toggle = (value: string) => {
    onChange(selected.includes(value) ? selected.filter((v) => v !== value) : [...selected, value]);
  };

  return (
    <div className="fsection">
      <span className="flabel">{label}</span>
      <div className="chip-toggle-group">
        {options.map((opt) => (
          <button
            key={opt}
            type="button"
            className={`chip-toggle${selected.includes(opt) ? " on" : ""}`}
            onClick={() => toggle(opt)}
          >
            {opt}
          </button>
        ))}
      </div>
    </div>
  );
}

const RECENCY_LABELS: Record<RecencyFilter, string> = {
  all: "Any time",
  "1": "Last 24 hours",
  "3": "Last 3 days",
  "7": "Last 7 days",
  "30": "Last 30 days",
};

/**
 * What's filtering the list, one removable chip each (asked 2026-11-06), for every filter but the
 * Interest and Fit toggles, which show their state where they are. Closed dropdowns hid it.
 */
function AppliedFilters({
  filters,
  onChange,
}: {
  filters: JobFilterState;
  onChange: (f: JobFilterState) => void;
}) {
  const chips: { key: string; label: string; clear: () => void }[] = [];
  const set = (patch: Partial<JobFilterState>) => onChange({ ...filters, ...patch });
  const search = filters.search.trim();
  if (search) chips.push({ key: "search", label: `Search: ${search}`, clear: () => set({ search: "" }) });
  const lists: [keyof Pick<JobFilterState, "organization" | "source" | "location">, string][] = [
    ["organization", "Organization"],
    ["source", "Source"],
    ["location", "Location"],
  ];
  for (const [field, name] of lists) {
    for (const value of filters[field]) {
      chips.push({
        key: `${field}:${value}`,
        label: `${name}: ${value}`,
        clear: () => set({ [field]: filters[field].filter((v) => v !== value) }),
      });
    }
  }
  if (filters.recency !== "all") {
    chips.push({
      key: "recency",
      label: `Posted: ${RECENCY_LABELS[filters.recency]}`,
      clear: () => set({ recency: "all" }),
    });
  }
  const minSalary = Number(filters.minSalary);
  if (filters.minSalary.trim() && minSalary > 0) {
    const amount = minSalary >= 1000 ? `$${Math.round(minSalary / 1000)}k` : `$${minSalary}`;
    chips.push({ key: "salary", label: `Salary: ${amount}+`, clear: () => set({ minSalary: "" }) });
  }
  if (!filters.includeUnscored) {
    chips.push({ key: "unscored", label: "Not yet scored: hidden", clear: () => set({ includeUnscored: true }) });
  }
  if (filters.includeDealbreakers) {
    chips.push({
      key: "dealbreakers",
      label: "Showing dealbreaker jobs",
      clear: () => set({ includeDealbreakers: false }),
    });
  }
  if (filters.onlyScreenedOut) {
    chips.push({
      key: "only-screened",
      label: "Only low-relevance jobs",
      clear: () => set({ onlyScreenedOut: false, includeScreenedOut: false }),
    });
  } else if (filters.includeScreenedOut) {
    chips.push({
      key: "screened",
      label: "Showing low-relevance jobs",
      clear: () => set({ includeScreenedOut: false }),
    });
  }
  if (chips.length === 0) return null;
  return (
    <div className="applied-filters" aria-label="Applied filters">
      {chips.map((c) => (
        <button key={c.key} type="button" className="table-filter-pill" onClick={c.clear} title="Remove this filter">
          <span className="applied-filter-text">{c.label}</span>
          <span aria-hidden="true">×</span>
        </button>
      ))}
    </div>
  );
}

function DropdownMultiSelect({
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

  // The selections themselves, not a count (asked 2026-11-06); a long list ends in an ellipsis (CSS).
  const summary = selected.length === 0 ? "All" : selected.join(", ");

  return (
    <div className="fsection">
      <span className="flabel">{label}</span>
      <details className="ms-dropdown">
        <summary>
          <span className="ms-dropdown-summary" title={summary}>
            {summary}
          </span>
        </summary>
        <div className="ms-dropdown-options">
          {options.map((opt) => (
            <label key={opt} className="multiselect-option">
              <input type="checkbox" checked={selected.includes(opt)} onChange={() => toggle(opt)} />
              <span>{opt}</span>
            </label>
          ))}
        </div>
      </details>
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
  const ruledOut = jobs.filter((j) => j.dealbreakers?.length).length;
  const setAside = jobs.filter((j) => j.screened_out).length;
  const organizations = Array.from(new Set(jobs.map((j) => j.company).filter(Boolean))).sort();
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
          // Truly clears, the default High/Med fit filters included (asked 2026-11-06); the sort stays.
          <button className="link small" onClick={() => onChange({ ...NO_JOB_FILTERS, sort: filters.sort })}>
            Clear all
          </button>
        )}
      </div>

      <AppliedFilters filters={filters} onChange={onChange} />

      <div className="fsection">
        <span className="flabel">Search</span>
        <input
          placeholder="Title, company, location, full description..."
          value={filters.search}
          onChange={(e) => set("search", e.target.value)}
          style={{ width: "100%" }}
        />
      </div>

      <ChipMultiSelect
        label="Interest fit"
        options={FIT_OPTIONS}
        selected={filters.interestFilter}
        onChange={(v) => set("interestFilter", v)}
      />

      <ChipMultiSelect
        label="Fit"
        options={FIT_OPTIONS}
        selected={filters.candidateFilter}
        onChange={(v) => set("candidateFilter", v)}
      />

      {/* Plain labels, each with what it means (2026-11-12: a tester didn't know what "scoring" or
          the "title screen", now the relevance check, were). */}
      <div className="fsection filter-checks">
        <label className="filter-check">
          <input
            type="checkbox"
            checked={filters.includeUnscored}
            onChange={(e) => set("includeUnscored", e.target.checked)}
          />
          <span>
            Include jobs not rated yet
            <span className="filter-check-hint">
              Each job is rated for interest and fit once; new ones wait for the next daily check.
            </span>
          </span>
        </label>
        {ruledOut > 0 && (
          <label className="filter-check">
            <input
              type="checkbox"
              checked={filters.includeDealbreakers}
              onChange={(e) => set("includeDealbreakers", e.target.checked)}
            />
            <span>
              Show {ruledOut} job{ruledOut === 1 ? "" : "s"} outside your locations or pay floor
              <span className="filter-check-hint">Your dealbreakers, set on the Profile page.</span>
            </span>
          </label>
        )}
        {setAside > 0 && (
          <label className="filter-check">
            <input
              type="checkbox"
              checked={filters.includeScreenedOut}
              onChange={(e) => set("includeScreenedOut", e.target.checked)}
            />
            <span>
              Show {setAside} low-relevance job{setAside === 1 ? "" : "s"}
              <span className="filter-check-hint">
                A quick check of their titles set these aside as clearly not what you're looking for.
              </span>
            </span>
          </label>
        )}
      </div>

      <DropdownMultiSelect
        label="Organization"
        options={organizations}
        selected={filters.organization}
        onChange={(v) => set("organization", v)}
      />

      <DropdownMultiSelect
        label="Source"
        options={sources}
        selected={filters.source}
        onChange={(v) => set("source", v)}
      />

      <div className="fsection">
        <span className="flabel">Posted</span>
        <div className="select-wrap">
          <select
            value={filters.recency}
            onChange={(e) => set("recency", e.target.value as RecencyFilter)}
          >
            <option value="all">Any time</option>
            <option value="1">Last 24 hours</option>
            <option value="3">Last 3 days</option>
            <option value="7">Last 7 days</option>
            <option value="30">Last 30 days</option>
          </select>
        </div>
      </div>

      <DropdownMultiSelect
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
    </div>
  );
}

/** A chip click on a job card: adds that value to the filter, or takes it off if it's already on. */
export function toggleFilterValue(
  f: JobFilterState,
  key: "location" | "organization",
  value: string,
): JobFilterState {
  const current = f[key];
  return { ...f, [key]: current.includes(value) ? current.filter((v) => v !== value) : [...current, value] };
}
