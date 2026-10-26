import { useEffect, useRef, useState } from "react";
import { api, ApiError, ResumeSummary } from "../api";

export default function ResumesPage() {
  const [resumes, setResumes] = useState<ResumeSummary[]>([]);
  const [profile, setProfile] = useState<string | null>(null);
  const [kind, setKind] = useState<"resume" | "cover_letter">("resume");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const refresh = () => {
    api.listResumes().then(setResumes);
    api
      .getCandidateProfile()
      .then((r) => setProfile(r.text))
      .catch((e) => {
        if (e instanceof ApiError && e.status === 404) setProfile(null);
      });
  };

  useEffect(refresh, []);

  const handleUpload = async () => {
    const file = fileInput.current?.files?.[0];
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      await api.uploadResume(file, kind);
      if (fileInput.current) fileInput.current.value = "";
      refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const handleRebuild = async () => {
    setBusy(true);
    setError(null);
    try {
      const r = await api.rebuildCandidateProfile();
      setProfile(r.text);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <h1>Resumes &amp; cover letters</h1>

      <div className="card">
        <h2>Upload</h2>
        <input ref={fileInput} type="file" accept=".pdf,.docx,.md,.txt" />
        <select value={kind} onChange={(e) => setKind(e.target.value as typeof kind)}>
          <option value="resume">Resume</option>
          <option value="cover_letter">Cover letter</option>
        </select>
        <button onClick={handleUpload} disabled={busy}>
          Upload
        </button>
        {error && <p style={{ color: "crimson" }}>{error}</p>}
      </div>

      <div className="card">
        <h2>Uploaded documents</h2>
        {resumes.length === 0 && <p className="muted">None yet.</p>}
        <ul>
          {resumes.map((r) => (
            <li key={r.id}>
              [{r.kind}] {r.file_name} — {r.text_length} chars —{" "}
              <button onClick={() => api.deleteResume(r.id).then(refresh)}>delete</button>
            </li>
          ))}
        </ul>
      </div>

      <div className="card">
        <h2>Candidate profile</h2>
        <button onClick={handleRebuild} disabled={busy || resumes.length === 0}>
          Rebuild from uploaded documents
        </button>
        {profile ? (
          <p style={{ whiteSpace: "pre-wrap" }}>{profile}</p>
        ) : (
          <p className="muted">No profile yet — upload at least one document and rebuild.</p>
        )}
      </div>
    </div>
  );
}
