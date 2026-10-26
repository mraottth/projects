import { useEffect, useRef, useState } from "react";
import { api, ApiError, ResumeSummary, Settings } from "../api";
import { Markdown } from "../components/Markdown";
import { Spinner } from "../components/Spinner";
import { TagInput } from "../components/TagInput";

function ResumesSection() {
  const [resumes, setResumes] = useState<ResumeSummary[]>([]);
  const [profile, setProfile] = useState<string | null>(null);
  const [kind, setKind] = useState<"resume" | "cover_letter">("resume");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showArchived, setShowArchived] = useState(false);
  const [libraryPath, setLibraryPath] = useState<string | null>(null);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);

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

  const current = resumes.filter((r) => !r.is_archived);
  const archived = resumes.filter((r) => r.is_archived);
  const renderRow = (r: ResumeSummary) => (
    <li key={r.id}>
      <span className="tag">{r.kind === "resume" ? "resume" : "cover letter"}</span>{" "}
      {!!r.is_template && <span className="star">★ </span>}
      {r.file_name} <span className="muted small">— {r.text_length.toLocaleString()} chars</span> —{" "}
      <button className="link small" onClick={() => api.deleteResume(r.id).then(refresh)}>
        delete
      </button>
    </li>
  );

  return (
    <>
      <div className="card">
        <h2>Resumes &amp; cover letters</h2>
        <div className="form-row">
          <input ref={fileInput} type="file" accept=".pdf,.docx,.md,.txt" />
          <select value={kind} onChange={(e) => setKind(e.target.value as typeof kind)}>
            <option value="resume">Resume</option>
            <option value="cover_letter">Cover letter</option>
          </select>
          <button className="primary" onClick={handleUpload} disabled={busy}>
            Upload
          </button>
        </div>
        {error && <p className="error-text small">{error}</p>}
        {libraryPath && (
          <p className="muted small">
            Bulk-imported from <code className="path">{libraryPath}</code> — new tailored docs save to{" "}
            <code className="path">{libraryPath}/AI Generated</code>.
          </p>
        )}

        <ul>{current.map(renderRow)}</ul>
        {resumes.length === 0 && <p className="empty-state">None yet.</p>}
        {archived.length > 0 && (
          <>
            <button className="link small collapsible-toggle" onClick={() => setShowArchived((s) => !s)}>
              {showArchived ? "Hide" : "Show"} {archived.length} archived document
              {archived.length === 1 ? "" : "s"}
            </button>
            {showArchived && <ul>{archived.map(renderRow)}</ul>}
          </>
        )}
      </div>

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

      <div className="fsection" style={{ marginBottom: 0 }}>
        <span className="flabel">Minimum salary</span>
        <input
          type="number"
          defaultValue={settings.salary_floor ?? ""}
          onBlur={(e) => save({ salary_floor: e.target.value ? Number(e.target.value) : null })}
          style={{ width: "200px" }}
        />
      </div>

      {busy && <p className="muted small">Saving...</p>}
      {saved && !busy && <p className="muted small">Saved.</p>}
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
    </div>
  );
}
