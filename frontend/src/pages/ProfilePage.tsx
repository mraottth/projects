import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, ApiError, GeneratedDocWithJob, ResumeSummary, Settings } from "../api";
import { Markdown } from "../components/Markdown";
import { Spinner } from "../components/Spinner";
import { TagInput } from "../components/TagInput";

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
  const [profile, setProfile] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [libraryPath, setLibraryPath] = useState<string | null>(null);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");

  const refresh = () => {
    api.listResumes().then(setResumes);
    api
      .getCandidateProfile()
      .then((r) => setProfile(r.text))
      .catch((e) => {
        if (e instanceof ApiError && e.status === 404) setProfile(null);
      });
    api
      .getSettings()
      .then((s) => setLibraryPath((s as unknown as Record<string, string>).resume_library_path ?? null));
  };

  useEffect(refresh, []);

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

  const startEditing = () => {
    setDraft(profile ?? "");
    setEditing(true);
  };

  const handleSaveEdit = async () => {
    setBusy(true);
    setError(null);
    try {
      const r = await api.updateCandidateProfile(draft);
      setProfile(r.text);
      setEditing(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

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

      <div className="card">
        <h2>Candidate profile</h2>
        <p className="muted small">Synthesized from everything uploaded above.</p>
        {!editing && (
          <div className="form-actions" style={{ marginTop: 0 }}>
            <button className="ghost" onClick={handleRebuild} disabled={busy || resumes.length === 0}>
              {busy ? <Spinner label="Rebuilding..." /> : "Rebuild from uploaded documents"}
            </button>
            {profile && (
              <button className="link small" onClick={startEditing} disabled={busy}>
                Edit
              </button>
            )}
          </div>
        )}
        {error && <p className="error-text small">{error}</p>}

        {editing ? (
          <div className="card-section">
            <textarea
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              rows={14}
              style={{ width: "100%", fontFamily: "inherit" }}
            />
            <div className="form-actions">
              <button className="primary" onClick={handleSaveEdit} disabled={busy || !draft.trim()}>
                {busy ? <Spinner label="Saving..." /> : "Save"}
              </button>
              <button className="ghost" onClick={() => setEditing(false)} disabled={busy}>
                Cancel
              </button>
            </div>
          </div>
        ) : profile ? (
          <div className="card-section">
            <Markdown text={profile} />
          </div>
        ) : (
          <p className="empty-state">No profile yet — upload at least one document and rebuild.</p>
        )}
      </div>
    </>
  );
}

function DesiredRoleSection() {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [busy, setBusy] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    api.getSettings().then(setSettings);
  }, []);

  if (!settings) return <p className="muted">Loading...</p>;

  const save = async (patch: Partial<Settings>) => {
    setBusy(true);
    setSaved(false);
    try {
      const updated = await api.putSettings(patch);
      setSettings(updated);
      setSaved(true);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card">
      <h2>What are you looking for?</h2>
      <p className="muted small">
        This drives both scoring (jobs that don't match get down-weighted) and company/job-board
        discovery on the Manage Sources page.
      </p>

      <div className="fsection">
        <span className="flabel">Desired job titles</span>
        <TagInput
          value={settings.prefilter_titles_include}
          onChange={(tags) => save({ prefilter_titles_include: tags })}
          placeholder="e.g. Senior Backend Engineer, Staff Engineer"
        />
      </div>

      <div className="fsection">
        <span className="flabel">Desired sectors</span>
        <TagInput
          value={settings.prefilter_sectors}
          onChange={(tags) => save({ prefilter_sectors: tags })}
          placeholder="e.g. Climate tech, Healthcare"
        />
      </div>

      <div className="fsection">
        <span className="flabel">Desired locations</span>
        <TagInput
          value={settings.prefilter_locations}
          onChange={(tags) => save({ prefilter_locations: tags })}
          placeholder="e.g. New York, Remote"
        />
      </div>

      <div className="fsection">
        <label className="field inline" style={{ marginBottom: 0 }}>
          <input
            type="checkbox"
            defaultChecked={settings.prefilter_remote_only}
            onChange={(e) => save({ prefilter_remote_only: e.target.checked })}
          />
          <span style={{ margin: 0, fontWeight: 400, color: "var(--ink)" }}>Remote only</span>
        </label>
      </div>

      <div className="fsection">
        <span className="flabel">Minimum salary</span>
        <input
          type="number"
          defaultValue={settings.salary_floor ?? ""}
          onBlur={(e) => save({ salary_floor: e.target.value ? Number(e.target.value) : null })}
          style={{ width: "200px" }}
        />
      </div>

      <div className="fsection" style={{ marginBottom: 0 }}>
        <span className="flabel">In your own words, what are you looking for?</span>
        <p className="muted small" style={{ marginTop: 0 }}>
          A few sentences about what you want — type of work, company stage, what you're trying to
          avoid, anything that doesn't fit neatly into the fields above. Used for both company/job-
          board discovery and job scoring, weighted heavily since it's your own description, not a
          structured guess.
        </p>
        <textarea
          defaultValue={settings.looking_for_notes}
          onBlur={(e) => save({ looking_for_notes: e.target.value })}
          rows={4}
          placeholder="e.g. I want to stay in early-stage startups doing hands-on technical work, ideally with a mission I care about. Not interested in pure management roles or big tech."
          style={{ width: "100%" }}
        />
      </div>

      {busy && <p className="muted small">Saving...</p>}
      {saved && !busy && <p className="muted small">Saved.</p>}
    </div>
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
      <DesiredRoleSection />
      <GeneratedDocumentsSection />
    </div>
  );
}
