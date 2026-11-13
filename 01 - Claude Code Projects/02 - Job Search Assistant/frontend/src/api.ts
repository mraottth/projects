import { DEMO, demoFetch, demoUrl } from "./demo";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

const NETWORK_ERROR = "Lost the connection to the app before it answered — check your connection and try again.";

// The app sits behind Google's sign-in (IAP), whose session runs out after a while away. A page left
// open then asked for data, got sent to Google's sign-in page (which a background request can't
// follow), and showed nothing until a reload (asked 2026-11-10). Marked as coming from a script,
// IAP answers 401 instead, so it's spotted, and App shows a banner (SessionExpiredBanner).
export const SESSION_EXPIRED_EVENT = "jobhunt:session-expired";
export const SESSION_EXPIRED_MESSAGE =
  "You've been signed out after some time away. Reload the page to sign back in — your data is safe.";

function sessionExpired(): never {
  window.dispatchEvent(new Event(SESSION_EXPIRED_EVENT));
  throw new ApiError(401, SESSION_EXPIRED_MESSAGE);
}

/** Every request the app makes goes through here (streams too). */
export async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  if (DEMO) return demoFetch(path, init); // the public demo answers from a recording (demo.ts)
  const headers = new Headers(init.headers);
  headers.set("X-Requested-With", "XMLHttpRequest");
  let res: Response;
  try {
    res = await fetch(path, { ...init, headers });
  } catch {
    // fetch() only rejects when no response arrived at all (worded "Load failed" in Safari, "Failed
    // to fetch" in Chrome). That's also what a request bounced to Google's sign-in looks like, when
    // IAP doesn't recognize it as a script's, so ask once more, plainly, before calling it a
    // network problem.
    if (await signedOut()) sessionExpired();
    throw new Error(NETWORK_ERROR);
  }
  if (res.status === 401) await refusedOrExpired(res);
  return res;
}

// A 401 is either IAP's (the Google sign-in ran out: reload) or this app's own server refusing the
// sign-in token it was handed (api/auth.py, `code: "signed_out"` with the reason), which no reload
// fixes. Calling both "signed out after time away" hid the real reason (caught live, 2026-11-11).
async function refusedOrExpired(res: Response): Promise<never> {
  const body = await res
    .clone()
    .json()
    .catch(() => null);
  if (body && body.code === "signed_out" && typeof body.detail === "string") {
    throw new ApiError(401, `The app couldn't confirm your Google sign-in (${body.detail}).`);
  }
  sessionExpired();
}

async function signedOut(): Promise<boolean> {
  try {
    const res = await fetch("/api/me", { headers: { "X-Requested-With": "XMLHttpRequest" } });
    if (res.status !== 401) return false;
    const body = await res.json().catch(() => null);
    return !(body && body.code === "signed_out"); // our own server's refusal isn't IAP's sign-out
  } catch {
    return false;
  }
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await apiFetch(path, init);
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail ?? res.statusText);
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

export interface ResumeSummary {
  id: number;
  kind: "resume" | "cover_letter";
  label: string | null;
  file_name: string;
  file_type: string;
  uploaded_at: string;
  is_archived: boolean | 0 | 1;
  is_template: boolean | 0 | 1;
  text_length: number;
}

export interface Job {
  id: number;
  title: string;
  company: string;
  location: string | null;
  remote_type: string | null;
  salary_min: number | null;
  salary_max: number | null;
  salary_currency: string | null;
  salary_period: string | null;
  description_text: string | null;
  url: string | null;
  posted_at: string | null;
  first_seen_at: string;
  needs_review: 0 | 1;
  // Pasting a job one of your sources already has tracks that job instead of adding a copy.
  already_in_sources?: boolean;
}

// Your LinkedIn network (D-036): Connections.csv, who to reach out to, and sources to consider.
export type NetworkTier = "High" | "Medium" | "Low";
export interface NetworkPerson {
  id: number;
  name: string;
  url: string | null;
  company: string | null;
  position: string | null;
  connected_on: string | null;
  why: string;
  job_id: number | null;
  job_title: string | null;
}
export interface NetworkOverview {
  uploaded_at: string | null;
  count: number;
  picks_error: string | null;
  tiers: Record<NetworkTier, NetworkPerson[]>;
  // Everyone else, each with why they aren't listed (so a missing person can be found).
  not_listed: NetworkPerson[];
  suggested_sources: { name: string; connections: number; tier: NetworkTier; why: string | null; leader: boolean }[];
}
export interface PersonAtJob {
  id: number;
  name: string;
  url: string | null;
  position: string | null;
  connected_on: string | null;
  tier: NetworkTier | null;
}

export type ApplicationStatus =
  | "saved"
  | "waiting"
  | "interviewing"
  | "offer"
  | "rejected_no_interview"
  | "rejected_interviewed"
  | "withdrawn";

// In pipeline order; My Jobs' status menu and filter chips use this order.
export const APPLICATION_STATUSES: ApplicationStatus[] = [
  "saved",
  "waiting",
  "interviewing",
  "offer",
  "rejected_no_interview",
  "rejected_interviewed",
  "withdrawn",
];

export const APPLICATION_STATUS_LABELS: Record<ApplicationStatus, string> = {
  saved: "Drafting",
  waiting: "Waiting",
  interviewing: "Interviewing",
  offer: "Offer",
  rejected_no_interview: "Rejected - no interview",
  rejected_interviewed: "Rejected - interviewed",
  withdrawn: "Withdrawn",
};

// My Jobs' "What's working?" analysis (core/app_insights.py), as last saved.
export interface ApplicationInsights {
  text: string;
  generated_at: string;
  applications: number;
  applied: number;
}

export interface Application {
  id: number;
  job_id: number;
  status: ApplicationStatus;
  applied_at: string | null;
  notes: string | null;
  next_action: string | null;
  next_action_date: string | null;
  title: string;
  company: string;
  location: string | null;
  url: string | null;
  salary_min: number | null;
  salary_max: number | null;
  salary_currency: string | null;
  salary_period: string | null;
  resume_doc_id: number | null;
  cover_letter_doc_id: number | null;
  history: { status: string; changed_at: string }[];
  updated_at: string; // SQLite datetime('now'): UTC without a zone marker
  // The job's latest score (null until it's scored).
  interest_fit: "High" | "Med" | "Low" | null;
  interest_score: number | null;
  candidate_fit: "High" | "Med" | "Low" | null;
  candidate_fit_score: number | null;
}

export interface ScoredJob extends Job {
  // How many of your LinkedIn connections work at its organization (D-036; 0 without an upload).
  connections?: number;
  // All Jobs only: which dealbreakers rule this job out ("location", "salary"); hidden unless asked.
  dealbreakers?: string[];
  // All Jobs only: the title screen set it aside as a clear mismatch, so it isn't scored.
  screened_out?: boolean;
  interest_fit: "High" | "Med" | "Low" | null;
  interest_score: number | null;
  candidate_fit: "High" | "Med" | "Low" | null;
  candidate_fit_score: number | null;
  summary_text: string | null;
  rationale_text: string | null;
  source_name: string | null;
  // The source is a job board (several employers), not one organization's own page (SQLite 0/1).
  source_is_board?: number | boolean | null;
  // All Jobs only: its status on My Jobs, or null when it isn't there.
  application_status?: ApplicationStatus | null;
}

export interface DigestEntry extends ScoredJob {
  rank: number;
}

export type RankedJob = ScoredJob;

export interface CompanySuggestion {
  company_name: string;
  reason: string;
  careers_url: string | null;
}

// Optional steering for one discovery search: free-text direction, and/or something to find more of.
export interface DiscoveryRequest {
  guidance?: string;
  like?: string;
}

const DISCOVERY_POLL_MS = 2000;
const MAX_POLL_FAILURES = 5;

// A discovery search takes a minute or two. Holding one request open that long got it dropped
// ("Load failed" in Safari), so it starts in the background and is polled with short requests.
async function runDiscovery<T>(startPath: string, req: DiscoveryRequest): Promise<T[]> {
  const { task_id } = await request<{ task_id: string }>(startPath, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  let failures = 0;
  for (;;) {
    await new Promise((resolve) => setTimeout(resolve, DISCOVERY_POLL_MS));
    let task: { status: "running" | "done" | "error"; result?: T[]; error?: string };
    try {
      task = await request(`/api/discovery/tasks/${task_id}`);
      failures = 0;
    } catch (e) {
      // A dropped poll (phone asleep, a network blip) doesn't lose the search; keep trying a bit.
      if (e instanceof ApiError || ++failures >= MAX_POLL_FAILURES) throw e;
      continue;
    }
    if (task.status === "done") return task.result ?? [];
    if (task.status === "error") throw new Error(task.error ?? "Discovery failed");
  }
}

export interface JobbyMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  created_at: string;
}

export interface JobbyThread {
  // null until its first message is sent (a new chat)
  thread_id: number | null;
  job: { id: number; title: string; company: string; status: ApplicationStatus | null } | null;
  messages: JobbyMessage[];
}

// One past chat on Jobby's home (GET /api/jobby/threads).
export interface JobbyChatSummary {
  id: number;
  job_id: number | null;
  // A general chat's title, written after its first reply (null until then, and for job chats).
  title: string | null;
  job_title: string | null;
  job_company: string | null;
  application_status: ApplicationStatus | null;
  message_count: number;
  preview: string;
  updated_at: string;
}

// Which conversation a message goes to: a chat reopened by id, a job's chat, or a new general one.
export type JobbyTarget = { threadId: number } | { jobId: number } | { newChat: true };

// One server-sent event from POST /api/jobby/thread/messages (see core/jobby.stream_reply).
export type JobbyEvent =
  | { type: "delta"; text: string }
  | { type: "status"; text: string }
  | { type: "done"; user: JobbyMessage; assistant: JobbyMessage; thread_id: number }
  | { type: "title"; thread_id: number; title: string }
  | { type: "error"; message: string };

function jobbyQuery(jobId: number | null) {
  return jobId === null ? "" : `?job_id=${jobId}`;
}

export interface JobBoardSuggestion {
  name: string;
  url: string;
  reason: string;
}

export type SourceKind =
  | "ats_greenhouse"
  | "ats_lever"
  | "ats_ashby"
  | "aggregator_adzuna"
  | "aggregator_remoteok"
  | "custom_board";

export interface Source {
  id: number;
  kind: SourceKind;
  name: string;
  company_name: string | null;
  config: Record<string, unknown>;
  enabled: boolean | 0 | 1;
  poll_frequency_hours: number;
  last_polled_at: string | null;
  last_success_at: string | null;
  last_error: string | null;
  created_at: string;
  job_count: number;
}

export interface Settings {
  daily_llm_cap: number;
  prefilter_keywords_include: string[];
  prefilter_keywords_exclude: string[];
  prefilter_titles_include: string[];
  prefilter_sectors: string[];
  prefilter_locations: string[];
  prefilter_remote_only: boolean;
  location_dealbreaker?: boolean;
  salary_dealbreaker?: boolean;
  salary_floor: number | null;
  looking_for_notes: string;
  onboarding_completed: boolean;
  onboarding_step: OnboardingStep;
  // Set when someone finishes or skips onboarding; the guided tour (Tour.tsx) runs once, then clears it.
  tour_pending?: boolean;
}

export type OnboardingStep =
  | "resume"
  | "looking_for"
  | "cover_letter"
  | "old_docs"
  | "profile"
  | "sources"
  | "finish";

export interface RunProgress {
  stage: "idle" | "starting" | "polling" | "scoring" | "done" | "failed";
  sources_done: number;
  sources_total: number;
  jobs_done: number;
  jobs_total: number;
  sources_pending?: string[];
  run_id: number | null;
  finished_run_id: number | null;
  last_summary: RunSummary | null;
  last_error: string | null;
}

export interface RunSummary {
  sources_polled?: number;
  jobs_fetched?: number;
  llm_scored_count?: number;
  errors?: string[];
  [key: string]: unknown;
}

// "Checked 4 of 12 sources...", "Scored 12 of 20 jobs..." -- the live label for a running check.
export function runProgressText(p: RunProgress | null, doneText = "Done"): string {
  if (!p) return "Starting...";
  if (p.stage === "polling") {
    if (!p.sources_total) return "Checking your sources for new postings...";
    const pending = p.sources_pending ?? [];
    const reading = pending.length
      ? ` Still reading ${pending.slice(0, 2).join(", ")}${pending.length > 2 ? ` +${pending.length - 2} more` : ""}.`
      : "";
    return `Checked ${p.sources_done} of ${p.sources_total} sources for new postings...${reading}`;
  }
  if (p.stage === "scoring") {
    return p.jobs_total ? `Scored ${p.jobs_done} of ${p.jobs_total} jobs...` : "Nothing new to score — wrapping up...";
  }
  if (p.stage === "done") return doneText;
  return "Starting...";
}

type RunEvent =
  | { type: "started"; run_id: number }
  | ({ type: "progress" } & RunProgress)
  | { type: "done"; summary: RunSummary }
  | { type: "error"; message: string };

// Reads a server-sent-event body (POST responses, which EventSource can't make) event by event.
async function readEvents<T>(res: Response, onEvent: (event: T) => void): Promise<void> {
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const chunk = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      for (const line of chunk.split("\n")) {
        if (line.startsWith("data: ")) onEvent(JSON.parse(line.slice(6)) as T);
      }
    }
  }
}

const RUN_POLL_MS = 3000;

// A full check of every source takes minutes. The server streams progress the whole time (so the
// connection isn't dropped as idle); if it drops anyway, the run carries on server-side and this
// follows it through /progress until it reports that run finished.
async function runDaily(
  cap: number | undefined,
  onProgress?: (p: RunProgress) => void,
  url = `/api/pipeline/run-daily/stream${cap ? `?cap=${cap}` : ""}`,
): Promise<RunSummary> {
  let runId: number | null = null;
  let result: RunSummary | null = null;
  let failure: string | null = null;
  try {
    const res = await apiFetch(url, { method: "POST" });
    if (!res.ok || !res.body) {
      const body = await res.json().catch(() => ({ detail: res.statusText }));
      throw new ApiError(res.status, typeof body.detail === "string" ? body.detail : res.statusText);
    }
    await readEvents<RunEvent>(res, (event) => {
      if (event.type === "started") runId = event.run_id;
      else if (event.type === "progress") onProgress?.(event);
      else if (event.type === "done") result = event.summary;
      else if (event.type === "error") failure = event.message;
    });
  } catch (e) {
    if (e instanceof ApiError || runId === null) throw e instanceof ApiError ? e : new Error(NETWORK_ERROR);
  }
  if (failure) throw new Error(failure);
  if (result) return result;

  // The stream ended early: keep following the run that's still going on the server.
  let misses = 0;
  for (;;) {
    await new Promise((resolve) => setTimeout(resolve, RUN_POLL_MS));
    let progress: RunProgress;
    try {
      progress = await request<RunProgress>("/api/pipeline/progress");
      misses = 0;
    } catch (e) {
      if (++misses >= 10) throw e;
      continue;
    }
    onProgress?.(progress);
    if (progress.finished_run_id !== null && runId !== null && progress.finished_run_id >= runId) {
      if (progress.finished_run_id === runId && progress.last_error) throw new Error(progress.last_error);
      return progress.finished_run_id === runId && progress.last_summary ? progress.last_summary : {};
    }
  }
}

/** New & Noteworthy's freshness line and Refresh button (GET /api/digest/status). */
export interface DigestStatus {
  last_updated: string | null;
  can_refresh: boolean;
  refresh_available_at: string | null;
  scored_today: number;
  daily_score_cap: number;
  over_budget: boolean;
}

export interface OnboardingStatus {
  needs_onboarding: boolean;
  step: OnboardingStep;
  has_resume: boolean;
  has_cover_letter: boolean;
  has_profile: boolean;
}

export interface GeneratedDoc {
  id: number;
  job_id: number;
  application_id: number | null;
  kind: "resume" | "cover_letter";
  based_on_resume_id: number | null;
  content_text: string;
  file_path: string | null;
  created_at: string;
}

export interface GeneratedDocWithJob extends GeneratedDoc {
  job_title: string;
  job_company: string;
}

export interface LlmCall {
  id: number;
  call_type: string;
  job_id: number | null;
  model: string;
  input_tokens: number;
  output_tokens: number;
  estimated_cost_usd: number | null;
  created_at: string;
  note: string | null;
  cache_read_tokens: number;
  cache_write_tokens: number;
}

/** The Usage log in plain terms (core/usage.py): by category, and calls gathered into activities. */
export type UsagePeriod = "day" | "week" | "month";
export interface UsageCall {
  id: number;
  created_at: string;
  model: string;
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  cost: number;
  detail: string;
}
export interface UsageSummary {
  period: UsagePeriod;
  total_cost: number;
  total_calls: number;
  categories: { key: string; label: string; cost: number; calls: number; amount: string }[];
  activities: {
    category: string;
    category_label: string;
    label: string;
    started_at: string;
    ended_at: string;
    cost: number;
    calls: UsageCall[];
  }[];
}

/** What the title screen has done (GET /api/settings/title-screen, core/screening.py). */
export interface TitleScreenInsight {
  screened: number;
  set_aside: number;
  kept: number;
  waiting: number;
  // Waiting jobs a location or pay dealbreaker hides: never scored, so never checked.
  skipped: number;
  last_screened_at: string | null;
  cost_this_month: number;
  calls_this_month: number;
  recent: { id: number; title: string; company: string; location: string | null; screened_at: string }[];
  // What each manual run would send and roughly cost; shown when over show_estimate_over.
  estimates: { waiting: { jobs: number; usd: number }; recheck: { jobs: number; usd: number } };
  show_estimate_over: number;
  running: boolean;
}

/** Who's signed in (GET /api/me, D-021). status "none" = never asked for access. */
export interface Me {
  email: string;
  status: "approved" | "pending" | "declined" | "blocked" | "none";
  is_admin: boolean;
  name: string | null;
  id?: number | null;
  /** This month's Claude spend and today's scoring, for Settings (approved people only). */
  usage?: {
    spend_this_month: number;
    monthly_budget_usd: number;
    remaining: number;
    over: boolean;
    exempt: boolean;
    scored_today: number;
    daily_score_cap: number;
  } | null;
}

/** The latest everyone run, for the Admin page's "Daily check" card (api/admin.py). */
export type ExperimentScore = {
  interest_fit: string;
  interest_score: number;
  candidate_fit: string;
  candidate_fit_score: number;
  rationale_text: string;
  cost_usd: number | null;
  output_tokens: number;
};

export type ExperimentArmSummary = {
  scored: number;
  failed: number;
  total_cost_usd: number;
  cost_per_job_usd: number | null;
  output_tokens_per_job: number | null;
  interested_rated_high_or_med: number | null;
  not_interested_rated_low: number | null;
  interviewed_fit_high_or_med: number | null;
};

/** The Admin page's scoring cost experiment (core/scoring_experiment.py). */
export type ScoringExperimentResult = {
  ran_at: string;
  jobs: {
    job_id: number;
    title: string;
    company: string;
    judgment: "interested" | "not_interested";
    interviewed: boolean;
    current: ExperimentScore | null;
    lean: ExperimentScore | null;
  }[];
  summary: {
    arms: { current: ExperimentArmSummary; lean: ExperimentArmSummary };
    agreement: {
      compared: number;
      interest_same: number | null;
      fit_same: number | null;
      interest_score_diff: number | null;
      fit_score_diff: number | null;
    };
    judged: { interested: number; not_interested: number; interviewed: number };
  };
};

export type RescoreScope = "my_jobs" | "all_jobs";

export type RescoreStatus = {
  scope: RescoreScope;
  count: number;
  estimate_usd: number;
  available_at: string | null;
  budget_remaining_usd: number | null;
  running: boolean;
};

export type RescoreResult = { scored: number; failed: number; stopped: string | null };

/** GET /api/settings/salary-check: what the minimum-salary dealbreaker hides (or would). */
export type SalaryCheck = {
  enabled: boolean;
  floor: number | null;
  preview: {
    total: number;
    with_pay: number;
    hidden: number;
    examples: { title: string; company: string; annual_low: number }[];
  };
};

/** GET /api/settings/location-check: how desired locations are read, and what they'd hide. */
export type LocationCheck = {
  enabled: boolean;
  entries: { entry: string; recognized: boolean; matches: string[] }[];
  preview: {
    total: number;
    hidden: number;
    hidden_locations: { location: string; count: number }[];
    kept: Record<string, number>;
    // Kept because the location couldn't be placed ("Hybrid", "Richmond", "3 locations").
    unsure_locations: { location: string; count: number }[];
  };
};

/** GET /api/settings/email-status: set up or not, and what the last check's email step did. */
export type EmailStatus = {
  configured: boolean;
  to: string | null;
  last?: {
    at: string;
    sent: boolean;
    count: number;
    to: string | null;
    reason: string | null;
    on_list_today: number;
    best_interest_today: string | null;
  } | null;
};

export interface DailyRunStatus {
  latest: {
    status: "running" | "ok" | "failed" | "unfinished";
    started_by: "scheduled" | "admin";
    started_at: string;
    finished_at: string | null;
    error: string | null;
    sources_polled: number | null;
    jobs_fetched: number | null;
    llm_scored_count: number | null;
    people: number | null;
    source_errors: string[];
  } | null;
  last_ok_finished_at: string | null;
}

export interface AdminUser {
  id: number;
  email: string;
  name: string | null;
  note: string | null;
  status: Me["status"];
  is_admin: 0 | 1;
  daily_score_cap: number;
  monthly_budget_usd: number;
  last_seen_at: string | null;
  created_at: string;
  decided_at: string | null;
  scored_today: number;
  spend_this_month: number;
  sources_followed: number;
  applications: number;
}

export const api = {
  getMe: () => request<Me>("/api/me"),

  requestAccess: (name: string) =>
    request<{ status: Me["status"] }>("/api/access/request", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }),

  listUsers: () =>
    request<{ users: AdminUser[]; shared_spend_this_month: number; daily_run: DailyRunStatus }>("/api/admin/users"),

  updateUser: (
    id: number,
    patch: Partial<Pick<AdminUser, "status" | "daily_score_cap" | "monthly_budget_usd">>,
  ) =>
    request<AdminUser>(`/api/admin/users/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    }),

  listResumes: () => request<ResumeSummary[]>("/api/resumes"),

  getJob: (id: number) => request<Job>(`/api/jobs/${id}`),

  createManualJob: (payload: { url?: string; raw_text?: string }) =>
    request<Job>("/api/jobs/manual", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  updateJob: (id: number, payload: Partial<Job>) =>
    request<Job>(`/api/jobs/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  confirmJob: (id: number) => request<Job>(`/api/jobs/${id}/confirm`, { method: "POST" }),

  // Scores one job now (about a cent; outside the daily cap, inside the monthly budget).
  scoreJob: (id: number) =>
    request<
      Pick<
        ScoredJob,
        "interest_fit" | "interest_score" | "candidate_fit" | "candidate_fit_score" | "summary_text" | "rationale_text"
      >
    >(`/api/jobs/${id}/score`, { method: "POST" }),

  getRankedJobs: () => request<RankedJob[]>("/api/jobs/ranked"),
  // {job id: description}, loaded after the list for All Jobs' search box.
  getRankedDescriptions: () => request<Record<string, string>>("/api/jobs/ranked/descriptions"),

  getDigestToday: () => request<{ date: string; entries: DigestEntry[] }>("/api/digest/today"),

  getDigestByDate: (date: string) =>
    request<{ date: string; entries: DigestEntry[] }>(`/api/digest/${date}`),

  getDigestDates: () => request<string[]>("/api/digest/dates"),

  submitFeedback: (jobId: number, interested: "interested" | "not_interested" | "maybe", note?: string) =>
    request<{ ok: true }>(`/api/jobs/${jobId}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ interested, note }),
    }),

  listApplications: () => request<Application[]>("/api/applications"),

  getApplicationInsights: () => request<ApplicationInsights | null>("/api/applications/insights"),

  buildApplicationInsights: () =>
    request<ApplicationInsights>("/api/applications/insights", { method: "POST" }),

  createApplication: (jobId: number, status: ApplicationStatus = "saved") =>
    request<Application>("/api/applications", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: jobId, status }),
    }),

  updateApplication: (
    id: number,
    payload: Partial<Pick<Application, "status" | "notes" | "applied_at">>,
  ) =>
    request<Application>(`/api/applications/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  deleteApplication: (id: number) =>
    request<{ ok: true }>(`/api/applications/${id}`, { method: "DELETE" }),

  getPreferenceProfile: () => request<{ text: string }>("/api/settings/preference-profile"),

  rebuildPreferenceProfile: () =>
    request<{ text: string }>("/api/settings/preference-profile/rebuild", { method: "POST" }),

  uploadResume: (
    file: File,
    kind: "resume" | "cover_letter",
    options?: { label?: string; isArchived?: boolean },
  ) => {
    const form = new FormData();
    form.append("file", file);
    form.append("kind", kind);
    if (options?.label) form.append("label", options.label);
    form.append("is_archived", options?.isArchived ? "true" : "false");
    return request<{ id: number; file_name: string; text_length: number }>(
      "/api/resumes/upload",
      { method: "POST", body: form },
    );
  },

  getNetwork: () => request<NetworkOverview>("/api/network"),

  uploadConnections: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<NetworkOverview>("/api/network/connections", { method: "POST", body: form });
  },

  deleteConnections: () => request<NetworkOverview>("/api/network/connections", { method: "DELETE" }),

  peopleAtJob: (jobId: number) => request<PersonAtJob[]>(`/api/network/job/${jobId}`),

  updateResume: (id: number, payload: { is_archived?: boolean; is_template?: boolean }) =>
    request<{ ok: true }>(`/api/resumes/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  deleteResume: (id: number) => request<{ ok: true }>(`/api/resumes/${id}`, { method: "DELETE" }),

  getCandidateProfile: () => request<{ text: string }>("/api/resumes/profile"),

  rebuildCandidateProfile: () =>
    request<{ text: string }>("/api/resumes/profile/rebuild", { method: "POST" }),

  updateCandidateProfile: (text: string) =>
    request<{ text: string }>("/api/resumes/profile", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    }),

  reviseCandidateProfile: (instructions: string) =>
    request<{ text: string }>("/api/resumes/profile/revise", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ instructions }),
    }),

  getOnboardingStatus: () => request<OnboardingStatus>("/api/onboarding/status"),

  // A few sector names from the current resume (one cheap call, then kept until the resume changes).
  getSectorSuggestions: () =>
    request<{ sectors: string[] }>("/api/onboarding/sector-suggestions", { method: "POST" }),

  generateStarterCoverLetter: () =>
    request<{ id: number; content_text: string }>("/api/onboarding/generate-cover-letter", {
      method: "POST",
    }),

  listSources: () => request<Source[]>("/api/sources"),

  createSource: (payload: Partial<Source> & { kind: SourceKind; name: string }) =>
    request<Source>("/api/sources", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  updateSource: (id: number, payload: Partial<Source>) =>
    request<Source>(`/api/sources/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  deleteSource: (id: number) => request<{ ok: true }>(`/api/sources/${id}`, { method: "DELETE" }),

  testSource: (id: number) =>
    request<{ count: number; preview: Record<string, unknown>[]; steps: string[] }>(`/api/sources/${id}/test`, {
      method: "POST",
    }),

  getSettings: () => request<Settings>("/api/settings"),

  getLocationCheck: () => request<LocationCheck>("/api/settings/location-check"),

  getSalaryCheck: () => request<SalaryCheck>("/api/settings/salary-check"),

  putSettings: (payload: Partial<Settings>) =>
    request<Settings>("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  getTitleScreen: () => request<TitleScreenInsight>("/api/settings/title-screen"),

  putBackScreened: (jobId: number) =>
    request<{ ok: true }>(`/api/settings/title-screen/${jobId}/keep`, { method: "POST" }),

  getEmailStatus: () => request<EmailStatus>("/api/settings/email-status"),

  sendTestEmail: () =>
    request<{ sent: boolean; count: number; to: string }>("/api/settings/test-email", { method: "POST" }),

  // Plain links (the browser downloads them): the whole database, and My Jobs as a spreadsheet.
  backupDownloadUrl: "/api/settings/backup",
  applicationsCsvUrl: DEMO ? demoUrl("downloads/my-jobs.csv") : "/api/applications/export.csv",

  listBackups: () => request<{ backups: { name: string; bytes: number }[] }>("/api/settings/backups"),

  deleteAccount: (confirm: string) =>
    request<{ backup: string | null }>("/api/settings/delete-account", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirm }),
    }),

  deleteUser: (id: number, confirm: string) =>
    request<{ backup: string | null }>(`/api/admin/users/${id}`, {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirm }),
    }),

  getRescoreStatus: (scope: RescoreScope) => request<RescoreStatus>(`/api/rescore/${scope}`),

  // Runs the relevance check now: unscreened waiting jobs, or with recheck every waiting one.
  runRelevanceCheck: async (recheck: boolean, onProgress: (done: number, total: number) => void) => {
    const res = await apiFetch(`/api/settings/title-screen/run?recheck=${recheck ? 1 : 0}`, { method: "POST" });
    if (!res.ok || !res.body) {
      let detail = res.statusText;
      try {
        detail = (await res.json()).detail ?? detail;
      } catch {
        // not JSON
      }
      throw new ApiError(res.status, detail);
    }
    let result: { screened: number; set_aside: number } | null = null;
    let failure: string | null = null;
    await readEvents<
      | { type: "progress"; done: number; total: number }
      | { type: "done"; screened: number; set_aside: number }
      | { type: "error"; message: string }
    >(res, (event) => {
      if (event.type === "progress") onProgress(event.done, event.total);
      else if (event.type === "done") result = { screened: event.screened, set_aside: event.set_aside };
      else failure = event.message;
    });
    if (failure) throw new Error(failure);
    if (!result) throw new Error(NETWORK_ERROR);
    return result as { screened: number; set_aside: number };
  },

  rescore: async (scope: RescoreScope, onProgress: (done: number, total: number) => void) => {
    const res = await apiFetch(`/api/rescore/${scope}`, { method: "POST" });
    if (!res.ok || !res.body) {
      let detail = res.statusText;
      try {
        detail = (await res.json()).detail ?? detail;
      } catch {
        // not JSON
      }
      throw new ApiError(res.status, detail);
    }
    let result: RescoreResult | null = null;
    let failure: string | null = null;
    await readEvents<
      | { type: "progress"; done: number; total: number }
      | ({ type: "done" } & RescoreResult)
      | { type: "error"; message: string }
    >(res, (event) => {
      if (event.type === "progress") onProgress(event.done, event.total);
      else if (event.type === "done") result = { scored: event.scored, failed: event.failed, stopped: event.stopped };
      else failure = event.message;
    });
    if (failure) throw new Error(failure);
    if (!result) throw new Error(NETWORK_ERROR);
    return result as RescoreResult;
  },

  getScoringExperiment: () =>
    request<{ sample_size: number; estimate_usd: number; result: ScoringExperimentResult | null; running: boolean }>(
      "/api/admin/scoring-experiment",
    ),

  runScoringExperiment: async (onProgress: (done: number, total: number) => void) => {
    const res = await apiFetch("/api/admin/scoring-experiment", { method: "POST" });
    if (!res.ok || !res.body) throw new ApiError(res.status, res.statusText);
    let result: ScoringExperimentResult | null = null;
    let failure: string | null = null;
    await readEvents<
      | { type: "progress"; done: number; total: number }
      | { type: "done"; result: ScoringExperimentResult }
      | { type: "error"; message: string }
    >(res, (event) => {
      if (event.type === "progress") onProgress(event.done, event.total);
      else if (event.type === "done") result = event.result;
      else failure = event.message;
    });
    if (failure) throw new Error(failure);
    if (!result) throw new Error(NETWORK_ERROR);
    return result as ScoringExperimentResult;
  },

  deleteAllData: (confirm: string) =>
    request<{ tables_cleared: number; files_deleted: number; backup: string | null }>("/api/settings/delete-all-data", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirm }),
    }),

  getUsageLog: (period: UsagePeriod = "week") =>
    request<{
      calls: LlmCall[];
      totals: { total_cost: number; today_cost: number; total_calls: number };
      // Last 7 days: share of input tokens read from the prompt cache, and the net saving.
      cache: { hit_rate: number; saved_usd: number };
      summary: UsageSummary;
    }>(`/api/settings/usage-log?period=${period}`),

  runDailyNow: (cap?: number, onProgress?: (p: RunProgress) => void) => runDaily(cap, onProgress),

  getDigestStatus: () => request<DigestStatus>("/api/digest/status"),

  // This person's own check, with a 30-minute cooldown (a 429 says how long is left).
  refreshDigest: (onProgress?: (p: RunProgress) => void) =>
    runDaily(undefined, onProgress, "/api/digest/refresh/stream"),

  getRunProgress: () => request<RunProgress>("/api/pipeline/progress"),

  generateResume: (jobId: number, applicationId?: number) =>
    request<{ id: number; content_text: string; file_path: string | null }>("/api/generate/resume", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: jobId, application_id: applicationId ?? null }),
    }),

  generateCoverLetter: (jobId: number, applicationId?: number) =>
    request<{ id: number; content_text: string; file_path: string | null }>(
      "/api/generate/cover-letter",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_id: jobId, application_id: applicationId ?? null }),
      },
    ),

  getGenerationHistory: (jobId: number) =>
    request<GeneratedDoc[]>(`/api/generate/history?job_id=${jobId}`),

  getAllGeneratedDocuments: () => request<GeneratedDocWithJob[]>("/api/generate/documents"),

  generatedDocumentDownloadUrl: (docId: number) =>
    DEMO ? demoUrl(`downloads/document-${docId}.docx`) : `/api/generate/documents/${docId}/download`,

  resumeDownloadUrl: (resumeId: number) =>
    DEMO ? demoUrl(`downloads/resume-${resumeId}.docx`) : `/api/resumes/${resumeId}/download`,

  getJobbyThread: (jobId: number | null) => request<JobbyThread>(`/api/jobby/thread${jobbyQuery(jobId)}`),
  getJobbyChat: (threadId: number) => request<JobbyThread>(`/api/jobby/thread?thread_id=${threadId}`),
  listJobbyChats: () => request<JobbyChatSummary[]>("/api/jobby/threads"),
  deleteJobbyChat: (threadId: number) => request<{ ok: true }>(`/api/jobby/threads/${threadId}`, { method: "DELETE" }),

  clearJobbyThread: (jobId: number | null) =>
    request<{ ok: true }>(`/api/jobby/thread${jobbyQuery(jobId)}`, { method: "DELETE" }),

  // Sends a message and calls onEvent for each streamed event as it arrives. The reply streams
  // as server-sent events over a POST, which EventSource can't do, so this reads the body directly.
  streamJobbyMessage: async (target: JobbyTarget, content: string, onEvent: (e: JobbyEvent) => void) => {
    const query =
      "threadId" in target ? `?thread_id=${target.threadId}` : "jobId" in target ? `?job_id=${target.jobId}` : "?new=true";
    const res = await apiFetch(`/api/jobby/thread/messages${query}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content }),
    });
    if (!res.ok || !res.body) {
      const body = await res.json().catch(() => ({ detail: res.statusText }));
      const detail = typeof body.detail === "string" ? body.detail : res.statusText;
      throw new ApiError(res.status, detail);
    }
    await readEvents<JobbyEvent>(res, onEvent);
  },

  suggestCompanies: (req: DiscoveryRequest = {}) =>
    runDiscovery<CompanySuggestion>("/api/discovery/suggest/start", req),

  suggestJobBoards: (req: DiscoveryRequest = {}) =>
    runDiscovery<JobBoardSuggestion>("/api/discovery/suggest-boards/start", req),

  resolveSource: (companyName: string) =>
    request<Source>("/api/sources/resolve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ company_name: companyName }),
    }),
};
