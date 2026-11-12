import { DEMO } from "../demo";
import { useEffect, useRef, useState } from "react";
import { JobLink } from "../components/JobLink";
import { api, type Application, GeneratedDocWithJob, ResumeSummary } from "../api";
import { CandidateProfileCard } from "../components/CandidateProfileCard";
import { DesiredRoleForm } from "../components/DesiredRoleForm";
import { Spinner } from "../components/Spinner";

// Layout (2026-11-04, at the user's ask): "What are you looking for?" first, on the left, since it's
// the part that needs your input; a narrower "My documents" card on the right holds the current
// resume and cover letter, with older documents collapsed underneath. The candidate profile and
// generated documents follow at full width.

/** Picks a file and uploads it at once, so a narrow card has no wide native file input. */
function UploadButton({
  label,
  onFile,
  busy,
}: {
  label: string;
  onFile: (file: File) => Promise<void>;
  busy: boolean;
}) {
  const input = useRef<HTMLInputElement>(null);
  return (
    <>
      <input
        ref={input}
        type="file"
        accept=".pdf,.docx,.md,.txt"
        hidden
        onChange={async (e) => {
          const file = e.target.files?.[0];
          if (file) await onFile(file);
          if (input.current) input.current.value = "";
        }}
      />
      <button className="ghost small" onClick={() => input.current?.click()} disabled={busy}>
        {busy ? <Spinner label="Uploading..." /> : label}
      </button>
    </>
  );
}

function CurrentDoc({
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
  const [busy, setBusy] = useState(false);

  const upload = async (file: File) => {
    setBusy(true);
    try {
      await api.uploadResume(file, kind);
      onChanged();
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="doc-slot">
      <h3>{kind === "resume" ? "Current resume" : "Current cover letter"}</h3>
      {doc ? (
        <div
          className="current-doc"
          draggable
          title='Drag to "Older documents" to retire it'
          onDragStart={(e) => e.dataTransfer.setData("text/plain", String(doc.id))}
        >
          <span className="doc-name">{doc.file_name}</span>
          <span className="muted small">{doc.text_length.toLocaleString()} chars</span>
        </div>
      ) : (
        <p className="empty-state small" style={{ margin: "2px 0" }}>
          None yet.
        </p>
      )}
      {doc && doc.file_type !== "docx" && (
        // Only a .docx can be tailored in place (asked 2026-11-11: say so where the file is).
        <p className="doc-hint small">
          {kind === "resume"
            ? "Tailored resumes keep your own layout only from a Word file. Upload the .docx version to get edits made right in it; from this one you get a plain Word file instead."
            : "Tailored cover letters keep your own layout only from a Word file. Upload the .docx version to get one written into it; for now this letter is used as a sample of how you write."}
        </p>
      )}
      <div className="doc-actions">
        {doc && (
          <a className="link small doc-download" href={api.resumeDownloadUrl(doc.id)} download>
            Download
          </a>
        )}
        {!DEMO && <UploadButton label={doc ? "Replace" : "Upload"} onFile={upload} busy={busy} />}
      </div>
    </div>
  );
}

function OlderDocs({
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

  const act = async (action: () => Promise<unknown>) => {
    try {
      await action();
      onChanged();
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e));
    }
  };

  const upload = async (file: File) => {
    setBusy(true);
    try {
      await api.uploadResume(file, kind, { isArchived: true });
      onChanged();
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      className={`doc-slot older-docs${dragOver ? " drop-target" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
    >
      <button className="link collapsible-toggle" onClick={() => setExpanded((s) => !s)} aria-expanded={expanded}>
        {expanded ? "▾" : "▸"} Older documents ({docs.length})
      </button>
      {!expanded && (
        <p className="muted small" style={{ margin: "4px 0 0" }}>
          Past resumes and cover letters help build your profile. Drag a current one here to retire it.
        </p>
      )}
      {expanded && (
        <>
          <ul className="older-doc-list">
            {docs.map((d) => (
              <li key={d.id}>
                <div className="doc-name">{d.file_name}</div>
                <div className="small">
                  <span className="tag">{d.kind === "resume" ? "resume" : "cover letter"}</span>{" "}
                  <a className="link small" href={api.resumeDownloadUrl(d.id)} download>
                    download
                  </a>{" "}
                  ·{" "}
                  <button className="link small" onClick={() => act(() => api.updateResume(d.id, { is_template: true }))}>
                    make current
                  </button>{" "}
                  ·{" "}
                  <button className="link small" onClick={() => act(() => api.deleteResume(d.id))}>
                    delete
                  </button>
                </div>
              </li>
            ))}
            {docs.length === 0 && <li className="empty-state small">Nothing here yet.</li>}
          </ul>
          <div className="doc-actions">
            <select value={kind} onChange={(e) => setKind(e.target.value as typeof kind)}>
              <option value="resume">Resume</option>
              <option value="cover_letter">Cover letter</option>
            </select>
            {!DEMO && <UploadButton label="Add an older one" onFile={upload} busy={busy} />}
          </div>
        </>
      )}
    </div>
  );
}

/**
 * Tailor the current resume or cover letter for one job on My Jobs, from My documents (asked
 * 2026-11-12). The same generation as My Jobs' Prep column and the job page's buttons.
 */
function TailorForJob({ onMade }: { onMade: () => void }) {
  const [apps, setApps] = useState<Application[] | null>(null);
  const [appId, setAppId] = useState<number | "">("");
  const [busy, setBusy] = useState<"resume" | "cover" | null>(null);
  const [made, setMade] = useState<{ id: number; kind: string } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .listApplications()
      .then(setApps)
      .catch(() => setApps([]));
  }, []);

  if (apps === null) return null;
  const app = apps.find((a) => a.id === appId);
  const make = async (kind: "resume" | "cover") => {
    if (!app) return;
    setBusy(kind);
    setError(null);
    setMade(null);
    try {
      const doc =
        kind === "resume"
          ? await api.generateResume(app.job_id, app.id)
          : await api.generateCoverLetter(app.job_id, app.id);
      setMade({ id: doc.id, kind: kind === "resume" ? "resume" : "cover letter" });
      onMade();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="doc-slot tailor-for-job">
      <h3>Tailor for a job</h3>
      {apps.length === 0 ? (
        <p className="muted small" style={{ margin: 0 }}>
          Add a job to My Jobs to tailor your documents for it.
        </p>
      ) : (
        <>
          <select
            value={appId}
            onChange={(e) => {
              setAppId(e.target.value ? Number(e.target.value) : "");
              setMade(null);
            }}
            aria-label="Job to tailor for"
          >
            <option value="">Choose a job on My Jobs...</option>
            {apps.map((a) => (
              <option key={a.id} value={a.id}>
                {a.title} — {a.company}
              </option>
            ))}
          </select>
          <div className="doc-actions">
            <button className="ghost small" disabled={!app || !!busy} onClick={() => make("resume")}>
              {busy === "resume" ? <Spinner label="Writing..." /> : "Tailor resume"}
            </button>
            <button className="ghost small" disabled={!app || !!busy} onClick={() => make("cover")}>
              {busy === "cover" ? <Spinner label="Writing..." /> : "Tailor cover letter"}
            </button>
          </div>
          {made && (
            <a className="link small doc-download" href={api.generatedDocumentDownloadUrl(made.id)} download>
              Download the tailored {made.kind}
            </a>
          )}
          {error && <p className="error-text small">{error}</p>}
        </>
      )}
    </div>
  );
}

function MyDocuments({
  resumes,
  onChanged,
  onGenerated,
}: {
  resumes: ResumeSummary[];
  onChanged: () => void;
  onGenerated: () => void;
}) {
  const [error, setError] = useState<string | null>(null);
  const [libraryPath, setLibraryPath] = useState<string | null>(null);

  useEffect(() => {
    api
      .getSettings()
      .then((s) => setLibraryPath((s as unknown as Record<string, string>).resume_library_path ?? null));
  }, []);

  const currentResume = resumes.find((r) => r.kind === "resume" && r.is_template);
  const currentCoverLetter = resumes.find((r) => r.kind === "cover_letter" && r.is_template);
  const older = resumes.filter((r) => r.id !== currentResume?.id && r.id !== currentCoverLetter?.id);
  const changed = () => {
    setError(null);
    onChanged();
  };

  return (
    <div className="card my-documents">
      <h2>My documents</h2>
      {error && <p className="error-text small">{error}</p>}
      <CurrentDoc kind="resume" doc={currentResume} onChanged={changed} onError={setError} />
      <CurrentDoc kind="cover_letter" doc={currentCoverLetter} onChanged={changed} onError={setError} />
      {!DEMO && currentResume && <TailorForJob onMade={onGenerated} />}
      <OlderDocs docs={older} onChanged={changed} onError={setError} />
      {libraryPath && (
        <p className="muted small" style={{ margin: "10px 0 0" }}>
          Bulk-imported from <code className="path">{libraryPath}</code>; tailored documents save to its{" "}
          <code className="path">AI Generated</code> folder.
        </p>
      )}
    </div>
  );
}

function GeneratedDocumentsSection({ version = 0 }: { version?: number }) {
  const [docs, setDocs] = useState<GeneratedDocWithJob[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .getAllGeneratedDocuments()
      .then(setDocs)
      .finally(() => setLoading(false));
  }, [version]);

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
                  <JobLink jobId={d.job_id}>
                    {d.job_title} — {d.job_company}
                  </JobLink>
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
  const [resumes, setResumes] = useState<ResumeSummary[]>([]);
  const refresh = () => {
    api.listResumes().then(setResumes);
  };
  useEffect(refresh, []);
  // Bumped when a document is tailored from My documents, so the Generated documents list reloads.
  const [generated, setGenerated] = useState(0);

  return (
    <div>
      <h1>Profile</h1>
      <p className="muted" style={{ marginBottom: "18px" }}>
        What you're looking for and who you are: it drives scoring, New &amp; Noteworthy and every
        tailored document. Company and job-board discovery lives on Manage Sources.
      </p>
      <div className="profile-top">
        <DesiredRoleForm />
        <MyDocuments resumes={resumes} onChanged={refresh} onGenerated={() => setGenerated((n) => n + 1)} />
      </div>
      <CandidateProfileCard hasDocuments={resumes.length > 0} />
      <GeneratedDocumentsSection version={generated} />
    </div>
  );
}
