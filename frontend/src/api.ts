export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
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
  history: { status: string; changed_at: string }[];
}

export interface DigestEntry extends Job {
  interest_fit: "High" | "Med" | "Low" | null;
  interest_score: number | null;
  candidate_fit: "High" | "Med" | "Low" | null;
  candidate_fit_score: number | null;
  summary_text: string | null;
  rationale_text: string | null;
  rank: number;
}

export const api = {
  listResumes: () => request<ResumeSummary[]>("/api/resumes"),

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

  updateApplication: (id: number, payload: Partial<Pick<Application, "status" | "notes">>) =>
    request<Application>(`/api/applications/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  getPreferenceProfile: () => request<{ text: string }>("/api/settings/preference-profile"),

  rebuildPreferenceProfile: () =>
    request<{ text: string }>("/api/settings/preference-profile/rebuild", { method: "POST" }),

  uploadResume: (file: File, kind: "resume" | "cover_letter", label?: string) => {
    const form = new FormData();
    form.append("file", file);
    form.append("kind", kind);
    if (label) form.append("label", label);
    return request<{ id: number; file_name: string; text_length: number }>(
      "/api/resumes/upload",
      { method: "POST", body: form },
    );
  },

  deleteResume: (id: number) => request<{ ok: true }>(`/api/resumes/${id}`, { method: "DELETE" }),

  getCandidateProfile: () => request<{ text: string }>("/api/resumes/profile"),

  rebuildCandidateProfile: () =>
    request<{ text: string }>("/api/resumes/profile/rebuild", { method: "POST" }),
};
