import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, GeneratedDocWithJob, ResumeSummary } from "../api";
import { CandidateProfileCard } from "../components/CandidateProfileCard";
import { DesiredRoleForm } from "../components/DesiredRoleForm";
import { Spinner } from "../components/Spinner";

function CurrentDocCard({
  kind,
  doc,
  onChanged,
  onError,
}: {
  kind: "resume" | "cover_letter";
  doc: ResumeSummary | undefined;
  onChanged: () => void;
  onError: (msg: string) => void;
}) {
  const fileInput = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);

  const upload = async () => {
    const file = fileInput.current?.files?.[0];
    if (!file) return;
    setBusy(true);
    try {
      await api.uploadResume(file, kind);
      if (fileInput.current) fileInput.current.value = "";
      onChanged();
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card">
      <h2>{kind === "resume" ? "Current resume" : "Current cover letter"}</h2>
      {doc ? (
        <div
          className="current-doc"
          draggable
          onDragStart={(e) => e.dataTransfer.setData("text/plain", String(doc.id))}
        >
          <span className="tag">{doc.file_name}</span>
          <span className="muted small"> — {doc.text_length.toLocaleString()} chars</span>
          <a className="link small doc-download" href={api.resumeDownloadUrl(doc.id)} download>
            Download
          </a>
        </div>
      ) : (
        <p className="empty-state">None yet — upload one below.</p>
      )}
      {doc && (
        <p className="muted small" style={{ marginTop: "6px" }}>
          Drag to "Old resumes and cover letters" below to replace it, or just upload a new one.
        </p>
      )}
      <div className="form-row" style={{ marginTop: "10px", marginBottom: 0 }}>
        <input ref={fileInput} type="file" accept=".pdf,.docx,.md,.txt" />
        <button className="ghost" onClick={upload} disabled={busy}>
          {busy ? <Spinner label="Uploading..." /> : doc ? "Replace" : "Upload"}
        </button>
      </div>
    </div>
  );
}

function OldDocsSection({
  docs,
  onChanged,
  onError,
}: {
  docs: ResumeSummary[];
  onChanged: () => void;
  onError: (msg: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [kind, setKind] = useState<"resume" | "cover_letter">("resume");
  const [busy, setBusy] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const idStr = e.dataTransfer.getData("text/plain");
    if (!idStr) return;
    try {
      await api.updateResume(Number(idStr), { is_archived: true });
      onChanged();
    } catch (e2) {
      onError(e2 instanceof Error ? e2.message : String(e2));
    }
  };

  const promote = async (id: number) => {
    try {
      await api.updateResume(id, { is_template: true });
      onChanged();
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e));
    }
  };

  const uploadToOld = async () => {
    const file = fileInput.current?.files?.[0];
    if (!file) return;
    setBusy(true);
    try {
      await api.uploadResume(file, kind, { isArchived: true });
      if (fileInput.current) fileInput.current.value = "";
      onChanged();
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      className={`card${dragOver ? " drop-target" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
    >
      <button className="link collapsible-toggle" onClick={() => setExpanded((s) => !s)}>
        {expanded ? "Hide" : "Show"} old resumes and cover letters ({docs.length})
      </button>
      {!expanded && (
        <p className="muted small" style={{ marginTop: "6px" }}>
          Drag a current file here to move it here instead.
        </p>
      )}
      {expanded && (
        <>
          <ul>
            {docs.map((d) => (
              <li key={d.id}>
                <span className="tag">{d.kind === "resume" ? "resume" : "cover letter"}</span>{" "}
                {d.file_name}{" "}
                <span className="muted small">— {d.text_length.toLocaleString()} chars</span> —{" "}
                <a className="link small" href={api.resumeDownloadUrl(d.id)} download>
                  download
                </a>{" "}
                <button className="link small" onClick={() => promote(d.id)}>
                  make current
                </button>{" "}
                <button className="link small" onClick={() => api.deleteResume(d.id).then(onChanged)}>
                  delete
                </button>
              </li>
            ))}
            {docs.length === 0 && <li className="empty-state">Nothing here.</li>}
          </ul>
          <div className="form-row" style={{ marginBottom: 0 }}>
            <input ref={fileInput} type="file" accept=".pdf,.docx,.md,.txt" />
            <select value={kind} onChange={(e) => setKind(e.target.value as typeof kind)}>
              <option value="resume">Resume</option>
              <option value="cover_letter">Cover letter</option>
            </select>
            <button className="ghost" onClick={uploadToOld} disabled={busy}>
              {busy ? <Spinner label="Uploading..." /> : "Add to Old"}
            </button>
          </div>
        </>
      )}
    </div>
  );
}

function ResumesSection() {
  const [resumes, setResumes] = useState<ResumeSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [libraryPath, setLibraryPath] = useState<string | null>(null);

  const refresh = () => {
    api.listResumes().then(setResumes);
    api
      .getSettings()
      .then((s) => setLibraryPath((s as unknown as Record<string, string>).resume_library_path ?? null));
  };

  useEffect(refresh, []);

  const currentResume = resumes.find((r) => r.kind === "resume" && r.is_template);
  const currentCoverLetter = resumes.find((r) => r.kind === "cover_letter" && r.is_template);
  const oldDocs = resumes.filter((r) => r.id !== currentResume?.id && r.id !== currentCoverLetter?.id);

  return (
    <>
      {libraryPath && (
        <p className="muted small">
          Bulk-imported from <code className="path">{libraryPath}</code> — new tailored docs save to{" "}
          <code className="path">{libraryPath}/AI Generated</code>.
        </p>
      )}
      {error && <p className="error-text small">{error}</p>}

      <CurrentDocCard kind="resume" doc={currentResume} onChanged={refresh} onError={setError} />
      <CurrentDocCard
        kind="cover_letter"
        doc={currentCoverLetter}
        onChanged={refresh}
        onError={setError}
      />
      <OldDocsSection docs={oldDocs} onChanged={refresh} onError={setError} />

      <CandidateProfileCard hasDocuments={resumes.length > 0} />
    </>
  );
}

function GeneratedDocumentsSection() {
  const [docs, setDocs] = useState<GeneratedDocWithJob[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .getAllGeneratedDocuments()
      .then(setDocs)
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="card">
      <h2>Generated documents</h2>
      <p className="muted small">
        Every resume and cover letter you've generated, linked back to the posting it was written
        for.
      </p>
      {loading && <p className="muted small">Loading...</p>}
      {!loading && docs.length === 0 && <p className="empty-state">None generated yet.</p>}
      {docs.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>Kind</th>
              <th>Job</th>
              <th>Created</th>
              <th>File</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {docs.map((d) => (
              <tr key={d.id}>
                <td>{d.kind === "resume" ? "Resume" : "Cover letter"}</td>
                <td>
                  <Link to={`/jobs/${d.job_id}`}>
                    {d.job_title} — {d.job_company}
                  </Link>
                </td>
                <td>{d.created_at}</td>
                <td>
                  {d.file_path ? (
                    <code className="path">{d.file_path}</code>
                  ) : (
                    <span className="muted small">text only</span>
                  )}
                </td>
                <td>
                  <a className="link small" href={api.generatedDocumentDownloadUrl(d.id)} download>
                    Download
                  </a>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export default function ProfilePage() {
  return (
    <div>
      <h1>Profile</h1>
      <p className="muted" style={{ marginBottom: "18px" }}>
        Who you are and what you're looking for — drives scoring and generation. Company/job-board
        discovery lives on the Manage Sources page.
      </p>
      <ResumesSection />
      <DesiredRoleForm />
      <GeneratedDocumentsSection />
    </div>
  );
}
