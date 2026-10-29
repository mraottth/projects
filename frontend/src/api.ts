export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

const NETWORK_ERROR = "Lost the connection to the app before it answered — check your connection and try again.";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch {
    // fetch() itself only rejects when no response arrived at all; browsers word that unhelpfully
    // ("Load failed" in Safari, "Failed to fetch" in Chrome).
    throw new Error(NETWORK_ERROR);
  }
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
}

export type ApplicationStatus =
  | "saved"
  | "applied"
  | "phone_screen"
  | "interview"
  | "offer"
  | "rejected"
  | "withdrawn";

export const APPLICATION_STATUS_LABELS: Record<ApplicationStatus, string> = {
  saved: "Drafting",
  applied: "Applied",
  phone_screen: "Phone screen",
  interview: "Interview",
  offer: "Offer",
  rejected: "Rejected",
  withdrawn: "Withdrawn",
};

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
}

export interface ScoredJob extends Job {
  interest_fit: "High" | "Med" | "Low" | null;
  interest_score: number | null;
  candidate_fit: "High" | "Med" | "Low" | null;
  candidate_fit_score: number | null;
  summary_text: string | null;
  rationale_text: string | null;
  source_name: string | null;
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
  job: { id: number; title: string; company: string; status: ApplicationStatus | null } | null;
  messages: JobbyMessage[];
}

// One server-sent event from POST /api/jobby/thread/messages (see core/jobby.stream_reply).
export type JobbyEvent =
  | { type: "delta"; text: string }
  | { type: "status"; text: string }
  | { type: "done"; user: JobbyMessage; assistant: JobbyMessage }
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
  salary_floor: number | null;
  looking_for_notes: string;
  onboarding_completed: boolean;
  onboarding_step: OnboardingStep;
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
}

export const api = {
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

  getRankedJobs: () => request<RankedJob[]>("/api/jobs/ranked"),

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
    request<{ count: number; preview: Record<string, unknown>[] }>(`/api/sources/${id}/test`, {
      method: "POST",
    }),

  getSettings: () => request<Settings>("/api/settings"),

  putSettings: (payload: Partial<Settings>) =>
    request<Settings>("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  getEmailStatus: () => request<{ configured: boolean; to: string | null }>("/api/settings/email-status"),

  sendTestEmail: () =>
    request<{ sent: boolean; count: number; to: string }>("/api/settings/test-email", { method: "POST" }),

  deleteAllData: (confirm: string) =>
    request<{ tables_cleared: number; files_deleted: number }>("/api/settings/delete-all-data", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirm }),
    }),

  getUsageLog: () =>
    request<{ calls: LlmCall[]; totals: { total_cost: number; today_cost: number; total_calls: number } }>(
      "/api/settings/usage-log",
    ),

  runDailyNow: (cap?: number) =>
    request<Record<string, unknown>>(
      cap ? `/api/pipeline/run-daily?cap=${cap}` : "/api/pipeline/run-daily",
      { method: "POST" },
    ),

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

  generatedDocumentDownloadUrl: (docId: number) => `/api/generate/documents/${docId}/download`,

  resumeDownloadUrl: (resumeId: number) => `/api/resumes/${resumeId}/download`,

  getJobbyThread: (jobId: number | null) => request<JobbyThread>(`/api/jobby/thread${jobbyQuery(jobId)}`),

  clearJobbyThread: (jobId: number | null) =>
    request<{ ok: true }>(`/api/jobby/thread${jobbyQuery(jobId)}`, { method: "DELETE" }),

  // Sends a message and calls onEvent for each streamed event as it arrives. The reply streams
  // as server-sent events over a POST, which EventSource can't do, so this reads the body directly.
  streamJobbyMessage: async (jobId: number | null, content: string, onEvent: (e: JobbyEvent) => void) => {
    const res = await fetch(`/api/jobby/thread/messages${jobbyQuery(jobId)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content }),
    });
    if (!res.ok || !res.body) {
      const body = await res.json().catch(() => ({ detail: res.statusText }));
      const detail = typeof body.detail === "string" ? body.detail : res.statusText;
      throw new ApiError(res.status, detail);
    }
    const reader = res.body.getReader();
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
          if (line.startsWith("data: ")) onEvent(JSON.parse(line.slice(6)) as JobbyEvent);
        }
      }
    }
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
