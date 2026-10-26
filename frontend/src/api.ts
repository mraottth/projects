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

export const api = {
  listResumes: () => request<ResumeSummary[]>("/api/resumes"),

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
